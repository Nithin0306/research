import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class WirelessSensorNetworkSimulator:
    
    def __init__(self, file_path: str, area_size: int = 100, p: float = 0.1,
                 init_energy: float = 2.0, freq: str = 'T'):
        # Initialize parameters
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
        
        # Initialize components
        self.df = None
        self.nodes = []
        self.num_nodes = 0
        self.rounds = 0
        
        # Try to load dataset
        try:
            self.df = self._load_dataset(file_path)
            self.nodes = self._initialize_nodes()
            self.num_nodes = len(self.nodes)
            self.rounds = self.df['round'].nunique()
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")
            print("Creating sample data for demonstration...")
            self._create_sample_data()
        
        # Statistics storage
        self.stats = {
            'alive_nodes': [], 'total_energy': [], 'chs_per_round': [], 'pdr': [],
            'avg_delay': [], 'throughput': [], 'energy_efficiency': [],
            'cluster_balance': [], 'first_death': None, 'last_death': None,
            'control_overhead': []
        }
    
    def _create_sample_data(self):
        """Create sample data when dataset is not available."""
        print("Creating sample sensor network data...")
        
        # Generate sample data
        num_sensors = 20
        num_rounds = 50
        
        data = []
        for round_num in range(num_rounds):
            for sensor_id in range(1, num_sensors + 1):
                # Simulate battery decay over time
                battery_level = max(0, 100 - (round_num * 2) - np.random.normal(0, 5))
                energy = battery_level / 100.0 * self.init_energy
                
                data.append({
                    'sensor_id': sensor_id,
                    'round': round_num,
                    'batterylevel': battery_level,
                    'energy': energy,
                    'date_d_m_y': f"01/01/2024",
                    'time': f"{round_num:02d}:00:00",
                    'temp_C': 20 + np.random.normal(0, 3),
                    'hpa_div_4': 250 + np.random.normal(0, 10),
                    'sensor_cycle': round_num % 10
                })
        
        self.df = pd.DataFrame(data)
        self.df['datetime'] = pd.to_datetime(self.df['date_d_m_y'] + ' ' + self.df['time'])
        self.nodes = self._initialize_nodes()
        self.num_nodes = len(self.nodes)
        self.rounds = self.df['round'].nunique()
        
        print(f"Sample data created: {len(self.df)} records, {self.num_nodes} nodes, {self.rounds} rounds")
    
    def _load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess the sensor dataset with proper field mapping."""
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with {len(df)} records")
            print(f"Available columns: {list(df.columns)}")
            
            # Validate required columns
            required_cols = ['sensor_id', 'date_d_m_y', 'time', 'batterylevel']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Parse datetime and create rounds
            df['datetime'] = pd.to_datetime(
                df['date_d_m_y'].astype(str) + ' ' + df['time'].astype(str),
                dayfirst=True,
                errors='coerce'
            )
            
            # Remove rows with invalid datetime
            df = df.dropna(subset=['datetime'])
            df = df.sort_values('datetime')
            
            # Create rounds based on time frequency
            df['round'] = df.groupby(pd.Grouper(key='datetime', freq=self.freq)).ngroup()
            
            # Convert battery level to energy (assuming batterylevel is in percentage)
            df['energy'] = df['batterylevel'] / 100.0 * self.init_energy
            
            # Add sensor cycle information if available
            if 'sensor_cycle' in df.columns:
                df['cycle'] = df['sensor_cycle']
            
            # Add environmental data for analysis
            if 'temp_C' in df.columns:
                df['temperature'] = df['temp_C']
            if 'hpa_div_4' in df.columns:
                df['pressure'] = df['hpa_div_4'] * 4  # Convert back to hPa
            
            print(f"Processed dataset: {len(df)} records across {df['round'].nunique()} rounds")
            print(f"Unique sensors: {df['sensor_id'].nunique()}")
            print(f"Time range: {df['datetime'].min()} to {df['datetime'].max()}")
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    class Node:
        """Represents a sensor node in the network."""
        __slots__ = ['id', 'x', 'y', 'energy', 'initial_energy', 'is_CH', 'cluster_head', 'alive',
                     'distance_to_bs', 'data_packets_sent', 'ctrl_packets_sent',
                     'packets_received', 'ch_history', 'last_CH_round', 'lifetime',
                     'cluster_member_count', 'sensor_type', 'real_energy_history']
        
        def __init__(self, node_id: int, x: float, y: float, initial_energy: float, sensor_type: str = 'default'):
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
            return np.hypot(self.x - other.x, self.y - other.y)
        
        def update_real_energy(self, real_energy: float) -> None:
            """Update node energy based on real dataset values."""
            self.real_energy_history.append(self.energy)
            self.energy = max(0, real_energy)  # Ensure non-negative energy
            if self.energy <= 0:
                self.alive = False
        
        def consume_tx(self, distance: float, packet_type: str = 'data') -> None:
            """Calculate transmission energy consumption."""
            if not self.is_alive():
                return
                
            pkt_size = self.data_packet_size if packet_type == 'data' else self.ctrl_packet_size
            d_threshold = 75
            
            if distance < d_threshold:
                energy = (self.tx_energy_per_bit + self.fs_energy * distance**2) * pkt_size
            else:
                energy = (self.tx_energy_per_bit + self.mp_energy * distance**4) * pkt_size
            
            self.energy = max(0, self.energy - energy)
            
            if packet_type == 'data':
                self.data_packets_sent += 1
            else:
                self.ctrl_packets_sent += 1
            
            if self.energy <= 0:
                self.alive = False
        
        def consume_rx(self, packet_type: str = 'data') -> None:
            """Calculate reception energy consumption."""
            if not self.is_alive():
                return
                
            pkt_size = self.data_packet_size if packet_type == 'data' else self.ctrl_packet_size
            energy = self.rx_energy_per_bit * pkt_size
            self.energy = max(0, self.energy - energy)
            self.packets_received += 1
            
            if self.energy <= 0:
                self.alive = False
        
        def consume_idle(self) -> None:
            """Calculate idle energy consumption."""
            if not self.is_alive():
                return
                
            self.energy = max(0, self.energy - self.idle_energy_per_round)
            if self.energy <= 0:
                self.alive = False
        
        def mark_participation(self, was_in_cluster: bool) -> None:
            """Track node participation metrics."""
            if self.is_alive():
                self.lifetime += 1
                if was_in_cluster:
                    self.cluster_member_count += 1
    
    def _initialize_nodes(self) -> List[Node]:
        """Initialize sensor nodes from the dataset."""
        nodes = []
        
        # Get unique sensors and their initial data
        unique_sensors = self.df['sensor_id'].unique()
        
        # Generate positions for each sensor (you can modify this based on your setup)
        np.random.seed(42)  # For reproducible results
        positions = np.random.uniform(0, self.area_size, size=(len(unique_sensors), 2))
        
        for idx, sensor_id in enumerate(unique_sensors):
            x, y = positions[idx]
            
            # Get initial energy and sensor type from first record
            sensor_data = self.df[self.df['sensor_id'] == sensor_id].iloc[0]
            initial_energy = sensor_data.get('energy', self.init_energy)
            sensor_type = sensor_data.get('sensor_type', 'default')
            
            node = self.Node(sensor_id, x, y, initial_energy, sensor_type)
            node.distance_to_bs = np.hypot(x - self.bs_x, y - self.bs_y)
            nodes.append(node)
        
        return nodes
    
    def _get_round_data(self, round_num: int) -> Dict:
        """Get sensor data for a specific round."""
        round_data = self.df[self.df['round'] == round_num]
        
        # Create a mapping of sensor_id to energy and other metrics
        sensor_metrics = {}
        for _, row in round_data.iterrows():
            sensor_id = row['sensor_id']
            sensor_metrics[sensor_id] = {
                'energy': max(0, row['energy']),  # Ensure non-negative energy
                'batterylevel': row['batterylevel'],
                'active': row['batterylevel'] > 0,  # Assume 0% battery means inactive
                'temperature': row.get('temperature', 0),
                'pressure': row.get('pressure', 0),
                'cycle': row.get('cycle', 0)
            }
        
        return sensor_metrics
    
    def _elect_cluster_heads(self, alive_nodes: List[Node], round_num: int, sensor_metrics: Dict) -> List[Node]:
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
            
            if np.random.random() < threshold:
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
    
    def _form_clusters(self, alive_nodes: List[Node], CHs: List[Node]) -> Tuple[
        Dict[int, List[Node]], int, int, int, List[float]]:
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
    
    def _data_transmission(self, CHs: List[Node], cluster_members: Dict[int, List[Node]],
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
            if ch.is_alive():
                ch.consume_tx(ch.distance_to_bs, 'data')
                total_data_tx += 1
        
        return total_data_tx, total_data_rx
    
    def _update_stats(self, round_num: int, alive_count: int, CHs: List[Node],
                     cluster_members: Dict[int, List[Node]], total_data_tx: int,
                     total_data_rx: int, total_ctrl_tx: int, delays: List[float]) -> None:
        """Update simulation statistics for the current round."""
        self.stats['alive_nodes'].append(alive_count)
        self.stats['total_energy'].append(sum(n.energy for n in self.nodes if n.is_alive()))
        self.stats['chs_per_round'].append(len(CHs))
        
        # Network performance metrics
        self.stats['pdr'].append(
            total_data_rx / total_data_tx if total_data_tx > 0 else 0
        )
        self.stats['avg_delay'].append(np.mean(delays) if delays else 0)
        self.stats['throughput'].append(total_data_rx)
        
        total_energy = self.stats['total_energy'][-1]
        self.stats['energy_efficiency'].append(
            total_data_rx * self.data_packet_size / total_energy
            if total_energy > 0 else 0
        )
        self.stats['control_overhead'].append(
            total_ctrl_tx / max(total_data_tx, 1)  # Avoid division by zero
        )
        
        # Cluster metrics
        if CHs:
            cluster_sizes = [len(cluster_members[ch.id]) for ch in CHs]
            self.stats['cluster_balance'].append(np.std(cluster_sizes) if cluster_sizes else 0)
        else:
            self.stats['cluster_balance'].append(0)
        
        # Track first and last deaths
        if alive_count < self.num_nodes and self.stats['first_death'] is None:
            self.stats['first_death'] = round_num + 1
    
    def _visualize_round(self, round_num: int, CHs: List[Node], 
                        cluster_members: Dict[int, List[Node]]) -> None:
        """Visualize the network state for the current round."""
        try:
            plt.figure(figsize=(12, 10))
            plt.title(f"LEACH Algorithm - Round {round_num+1}\nAlive: {sum(n.is_alive() for n in self.nodes)}/{self.num_nodes}")
            plt.xlim(0, self.area_size)
            plt.ylim(0, self.area_size + 20)
            plt.gca().set_aspect('equal')
            
            colors = plt.cm.tab20.colors
            
            for i, ch in enumerate(CHs):
                if not ch.is_alive():
                    continue
                color = colors[i % len(colors)]
                plt.plot(ch.x, ch.y, 'o', markersize=12, markeredgecolor='k',
                        color=color, label=f'CH {ch.id} (E={ch.energy:.3f}J)')
                plt.plot([ch.x, self.bs_x], [ch.y, self.bs_y], '--', color=color, alpha=0.7)
                
                for member in cluster_members.get(ch.id, []):
                    if member.is_alive():
                        plt.plot([member.x, ch.x], [member.y, ch.y], ':', color=color, alpha=0.5)
                        plt.plot(member.x, member.y, 'o', color=color, markersize=8, alpha=0.7)
            
            # Plot dead nodes and base station
            dead_x = [n.x for n in self.nodes if not n.is_alive()]
            dead_y = [n.y for n in self.nodes if not n.is_alive()]
            if dead_x:
                plt.plot(dead_x, dead_y, 'kx', markersize=10, label='Dead nodes')
            
            plt.plot(self.bs_x, self.bs_y, 'ks', markersize=16, label='Base Station')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def run_simulation(self, visualize: bool = False) -> None:
        """Run the complete simulation using real dataset."""
        print("Starting LEACH simulation with real dataset...")
        
        for round_num in range(self.rounds):
            # Get real sensor data for this round
            sensor_metrics = self._get_round_data(round_num)
            
            # Get alive nodes that are active in this round
            alive_nodes = [n for n in self.nodes if n.is_alive() and n.id in sensor_metrics 
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
            alive_count = sum(n.is_alive() for n in self.nodes)
            self._update_stats(
                round_num, alive_count, CHs, cluster_members,
                total_data_tx, total_data_rx, total_ctrl_tx, delays
            )
            
            # Visualization (optional)
            if visualize and round_num % 10 == 0:  # Show every 10th round
                self._visualize_round(round_num, CHs, cluster_members)
            
            # Progress update
            if round_num % 50 == 0:
                print(f"Round {round_num + 1}/{self.rounds} completed. Alive nodes: {alive_count}")
    
    def plot_metrics(self) -> None:
        """Plot the simulation metrics."""
        try:
            plt.figure(figsize=(18, 12))
            
            metrics = [
                ('alive_nodes', 'b', 'Alive Nodes', 'Count'),
                ('total_energy', 'r', 'Total Energy', 'Joules'),
                ('chs_per_round', 'g', 'Cluster Heads', 'Count'),
                ('pdr', 'm', 'Packet Delivery Ratio', 'Ratio'),
                ('avg_delay', 'c', 'Average Delay', 'seconds'),
                ('throughput', 'y', 'Throughput', 'Packets'),
                ('energy_efficiency', 'k', 'Energy Efficiency', 'bits/Joule'),
                ('cluster_balance', 'purple', 'Cluster Balance', 'Std Dev'),
                ('control_overhead', 'orange', 'Control Overhead', 'Ratio')
            ]
            
            for i, (metric, color, title, ylabel) in enumerate(metrics[:6]):
                plt.subplot(2, 3, i + 1)
                if self.stats[metric]:  # Check if data exists
                    plt.plot(self.stats[metric], f'{color}-o', markersize=4)
                plt.title(title)
                plt.xlabel('Round')
                plt.ylabel(ylabel)
                plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            # Additional plots for remaining metrics
            plt.figure(figsize=(15, 5))
            for i, (metric, color, title, ylabel) in enumerate(metrics[6:]):
                plt.subplot(1, 3, i + 1)
                if self.stats[metric]:  # Check if data exists
                    plt.plot(self.stats[metric], f'{color}-o', markersize=4)
                plt.title(title)
                plt.xlabel('Round')
                plt.ylabel(ylabel)
                plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Plotting error: {e}")
    
    def generate_summary(self) -> None:
        """Generate and print simulation summary."""
        try:
            # Node-level metrics
            node_data = []
            for node in self.nodes:
                stability_ratio = (
                    node.cluster_member_count / node.lifetime 
                    if node.lifetime > 0 else 0
                )
                node_data.append([
                    node.id, node.sensor_type, node.lifetime, node.ch_history,
                    node.data_packets_sent, node.packets_received,
                    f"{stability_ratio:.3f}", f"{node.energy:.3f}"
                ])
            
            summary_df = pd.DataFrame(node_data, columns=[
                'Node ID', 'Sensor Type', 'Lifetime (Rounds)', 'CH Count',
                'Data Sent', 'Data Received', 'Stability Ratio', 'Final Energy'
            ])
            
            print("\n" + "="*80)
            print("LEACH ALGORITHM PERFORMANCE ANALYSIS")
            print("="*80)
            print(f"Dataset Analysis Complete!")
            print(f"Total Nodes Analyzed: {self.num_nodes}")
            print(f"Total Rounds Simulated: {len(self.stats['alive_nodes'])}")
            
            print("\n=== Derived Node Metrics ===")
            print(summary_df.sort_values(by='Lifetime (Rounds)', ascending=False).to_string(index=False))
            
            # Network-level summary
            print("\n=== Network Lifetime Summary ===")
            print(f"Total Rounds: {len(self.stats['alive_nodes'])}")
            print(f"First Node Death: Round {self.stats['first_death'] or 'N/A'}")
            print(f"Last Node Death: Round {self.stats['last_death'] or 'N/A'}")
            
            if self.stats['alive_nodes']:
                print(f"Average Alive Nodes: {np.mean(self.stats['alive_nodes']):.2f}")
            if self.stats['chs_per_round']:
                print(f"Average Cluster Heads: {np.mean(self.stats['chs_per_round']):.2f}")
            
            print("\n=== Performance Metrics ===")
            if self.stats['pdr']:
                print(f"Average PDR: {np.mean(self.stats['pdr']):.4f}")
            if self.stats['throughput']:
                print(f"Average Throughput: {np.mean(self.stats['throughput']):.2f} packets/round")
            if self.stats['avg_delay']:
                print(f"Average Delay: {np.mean(self.stats['avg_delay']):.6f} seconds")
            if self.stats['energy_efficiency']:
                print(f"Energy Efficiency: {np.mean(self.stats['energy_efficiency']):.2f} bits/Joule")
            if self.stats['control_overhead']:
                print(f"Control Overhead: {np.mean(self.stats['control_overhead']):.4f} ctrl/data ratio")
            if self.stats['cluster_balance']:
                print(f"Cluster Balance (Std Dev): {np.mean(self.stats['cluster_balance']):.3f}")
            
            # Save results to file
            results_file = 'leach_analysis_results.csv'
            summary_df.to_csv(results_file, index=False)
            print(f"\nResults saved to: {results_file}")
            
        except Exception as e:
            print(f"Summary generation error: {e}")

# Example usage
if __name__ == "__main__":
    # Replace 'your_dataset.csv' with your actual dataset file path
    dataset_path = "sensor_data.csv"
    
    try:
        # Initialize the simulator
        simulator = WirelessSensorNetworkSimulator(
            file_path=dataset_path,
            area_size=100,
            p=0.1,
            init_energy=2.0,
            freq='H'  # Hourly rounds - adjust based on your data frequency
        )
        
        # Run simulation
        simulator.run_simulation(visualize=False)  # Set to True for visualization
        
        # Generate plots and summary
        simulator.plot_metrics()
        simulator.generate_summary()
        
    except Exception as e:
        print(f"Simulation error: {e}")
        print("Please check your dataset format and ensure all required columns are present.")