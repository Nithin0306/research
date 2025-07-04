#!/usr/bin/env python3
"""
Enhanced Adaptive Rotation Load Balancing (EARLB) Algorithm
Modified to work with real sensor dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import sys
import os

class EARLBNode:
    """EARLB Node class for dataset-based simulation"""
    
    def __init__(self, sensor_id, sensor_type, initial_battery=100):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.initial_battery = initial_battery
        self.current_battery = initial_battery
        self.is_cluster_head = False
        self.cluster_members = []
        self.parent_ch = None
        
        # Load balancing attributes
        self.current_load = 0.0
        self.processing_capacity = np.random.uniform(0.8, 1.5)
        self.communication_cost = np.random.uniform(0.1, 0.3)
        
        # Performance metrics
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_dropped = 0
        self.total_delay = 0.0
        self.response_times = []
        self.load_history = []
        self.energy_history = []
        
        # Energy consumption rates (adjusted for battery levels)
        self.tx_power_rate = 0.5    # Battery % per transmission
        self.rx_power_rate = 0.2    # Battery % per reception
        self.idle_power_rate = 0.01 # Battery % per idle cycle
        self.ch_extra_power = 0.3   # Additional CH power consumption
        
        # Sensor-specific attributes
        self.last_temp = 0.0
        self.last_pressure = 0.0
        self.sensor_cycle = 0
        self.is_active = True
        
    def update_from_sensor_data(self, temp_c, hpa_div_4, battery_level, sensor_cycle):
        """Update node state from sensor data"""
        self.last_temp = temp_c
        self.last_pressure = hpa_div_4 * 4  # Convert back to hPa
        self.current_battery = battery_level
        self.sensor_cycle = sensor_cycle
        
        # Node dies if battery too low
        if self.current_battery <= 5:
            self.is_active = False
            
        self.energy_history.append(self.current_battery)
        
    def calculate_data_priority(self):
        """Calculate data transmission priority based on sensor readings"""
        # Higher priority for extreme temperature readings
        temp_priority = 1.0
        if self.last_temp < 0 or self.last_temp > 40:
            temp_priority = 1.5
        elif self.last_temp < 5 or self.last_temp > 35:
            temp_priority = 1.3
            
        # Higher priority for abnormal pressure readings
        pressure_priority = 1.0
        if self.last_pressure < 950 or self.last_pressure > 1050:
            pressure_priority = 1.4
        elif self.last_pressure < 980 or self.last_pressure > 1020:
            pressure_priority = 1.2
            
        return temp_priority * pressure_priority
        
    def consume_energy(self, amount):
        """Consume battery energy"""
        self.current_battery = max(0, self.current_battery - amount)
        if self.current_battery <= 0:
            self.is_active = False
            
    def get_energy_ratio(self):
        """Get remaining energy ratio"""
        return self.current_battery / self.initial_battery if self.initial_battery > 0 else 0
        
    def calculate_ch_probability(self):
        """Calculate cluster head selection probability"""
        if not self.is_active or self.current_battery <= 10:
            return 0.0
            
        energy_factor = self.get_energy_ratio()
        load_factor = max(0.1, 1.0 - (self.current_load / 10.0))
        comm_factor = 1.0 - self.communication_cost
        
        # Bonus for certain sensor types
        type_bonus = 1.2 if self.sensor_type in ['temperature', 'pressure'] else 1.0
        
        return (energy_factor * 0.4 + load_factor * 0.3 + comm_factor * 0.2 + type_bonus * 0.1)
        
    def send_packet(self, packet_size=64):
        """Simulate packet transmission"""
        if not self.is_active:
            return False
            
        # Calculate energy consumption based on data priority
        priority = self.calculate_data_priority()
        energy_cost = self.tx_power_rate * (packet_size / 64.0) * priority
        
        if self.is_cluster_head:
            energy_cost += self.ch_extra_power
            
        self.consume_energy(energy_cost)
        
        if self.is_active:
            self.packets_sent += 1
            self.current_load += packet_size / 1000.0
            return True
        else:
            self.packets_dropped += 1
            return False
            
    def receive_packet(self, packet_size=64):
        """Simulate packet reception"""
        if not self.is_active:
            return False
            
        energy_cost = self.rx_power_rate * (packet_size / 64.0)
        
        if self.is_cluster_head:
            energy_cost += self.ch_extra_power
            
        self.consume_energy(energy_cost)
        
        if self.is_active:
            self.packets_received += 1
            self.current_load += packet_size / 2000.0  # CH processes more efficiently
            return True
        else:
            self.packets_dropped += 1
            return False
            
    def idle_cycle(self):
        """Simulate idle energy consumption"""
        if self.is_active:
            self.consume_energy(self.idle_power_rate)
            # Process current load
            processed = min(self.current_load, self.processing_capacity * 0.1)
            self.current_load = max(0, self.current_load - processed)
            self.load_history.append(self.current_load)

class EARLBDatasetSimulation:
    """EARLB simulation using real sensor dataset"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.nodes = {}
        self.sensor_data = None
        self.current_chs = set()
        self.ch_rotation_interval = 100  # Number of data points
        self.time_step = 0
        
        # Metrics storage (keeping same structure as original)
        self.metrics = {
            'packet_drop': [],
            'delay': [],
            'network_lifetime': [],
            'throughput': [],
            'energy_consumption': [],
            'response_time': [],
            'load_efficiency': [],
            'cluster_stability': [],
            'pdr': [],
            'scalability': [],
            'alive_nodes': [],
            'simulation_time': []
        }
        
        # Load and preprocess dataset
        self.load_dataset()
        self.initialize_nodes()
        
    def load_dataset(self):
        """Load and preprocess the sensor dataset"""
        print(f"Loading dataset from {self.dataset_path}...")
        
        try:
            # Read CSV file
            self.sensor_data = pd.read_csv(self.dataset_path)
            
            # Validate required columns
            required_cols = ['sort_id', 'date_d_m_y', 'time', 'sensor_id', 
                           'sensor_type', 'temp_C', 'hpa_div_4', 'batterylevel', 'sensor_cycle']
            
            missing_cols = [col for col in required_cols if col not in self.sensor_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Create datetime column
            self.sensor_data['datetime'] = pd.to_datetime(
                self.sensor_data['date_d_m_y'] + ' ' + self.sensor_data['time'],
                format='%d/%m/%Y %H:%M:%S'
            )
            
            # Sort by datetime and sort_id
            self.sensor_data = self.sensor_data.sort_values(['datetime', 'sort_id'])
            
            # Handle missing values
            self.sensor_data['temp_C'].fillna(self.sensor_data['temp_C'].mean(), inplace=True)
            self.sensor_data['hpa_div_4'].fillna(self.sensor_data['hpa_div_4'].mean(), inplace=True)
            self.sensor_data['batterylevel'].fillna(100, inplace=True)
            
            print(f"Dataset loaded successfully!")
            print(f"Total records: {len(self.sensor_data)}")
            print(f"Unique sensors: {self.sensor_data['sensor_id'].nunique()}")
            print(f"Time range: {self.sensor_data['datetime'].min()} to {self.sensor_data['datetime'].max()}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)
            
    def initialize_nodes(self):
        """Initialize nodes from dataset"""
        print("Initializing nodes from dataset...")
        
        # Get unique sensors
        unique_sensors = self.sensor_data[['sensor_id', 'sensor_type']].drop_duplicates()
        
        # Limit to 50 nodes as per original requirement
        if len(unique_sensors) > 50:
            unique_sensors = unique_sensors.head(50)
            print(f"Limited to first 50 sensors out of {len(unique_sensors)} available")
            
        # Create nodes
        for _, row in unique_sensors.iterrows():
            sensor_id = row['sensor_id']
            sensor_type = row['sensor_type']
            
            # Get initial battery level for this sensor
            initial_battery = self.sensor_data[
                self.sensor_data['sensor_id'] == sensor_id
            ]['batterylevel'].iloc[0]
            
            node = EARLBNode(sensor_id, sensor_type, initial_battery)
            self.nodes[sensor_id] = node
            
        print(f"Initialized {len(self.nodes)} nodes")
        
    def get_active_nodes(self):
        """Get list of active nodes"""
        return [node for node in self.nodes.values() if node.is_active]
        
    def select_cluster_heads(self):
        """EARLB cluster head selection algorithm"""
        active_nodes = self.get_active_nodes()
        
        if not active_nodes:
            return
            
        # Calculate optimal number of CHs
        optimal_ch_count = max(1, int(len(active_nodes) * 0.07))
        
        # Calculate probabilities
        candidates = []
        for node in active_nodes:
            prob = node.calculate_ch_probability()
            if prob > 0.1:
                candidates.append((node, prob))
                
        # Select CHs based on probability
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_chs = [node for node, _ in candidates[:optimal_ch_count]]
        
        # Reset CH status
        for node in self.nodes.values():
            node.is_cluster_head = False
            node.cluster_members = []
            
        # Set new CHs
        new_ch_set = set()
        for ch_node in selected_chs:
            ch_node.is_cluster_head = True
            new_ch_set.add(ch_node.sensor_id)
            
        # Form clusters
        self.form_clusters(selected_chs, active_nodes)
        
        # Update CH tracking
        self.current_chs = new_ch_set
        
    def form_clusters(self, cluster_heads, active_nodes):
        """Form clusters around selected cluster heads"""
        non_ch_nodes = [node for node in active_nodes if not node.is_cluster_head]
        
        for node in non_ch_nodes:
            best_ch = None
            best_score = float('inf')
            
            for ch_node in cluster_heads:
                # Calculate score based on energy, load, and sensor compatibility
                energy_factor = ch_node.get_energy_ratio()
                load_factor = max(0.1, 1.0 - (ch_node.current_load / 20.0))
                
                # Preference for same sensor type
                type_factor = 1.2 if node.sensor_type == ch_node.sensor_type else 1.0
                
                # Simulated distance factor
                distance_factor = np.random.uniform(0.5, 2.0)
                
                score = distance_factor / (energy_factor * load_factor * type_factor)
                
                if score < best_score:
                    best_score = score
                    best_ch = ch_node
                    
            if best_ch:
                node.parent_ch = best_ch.sensor_id
                best_ch.cluster_members.append(node.sensor_id)
                
    def process_time_step(self, time_step_data):
        """Process one time step of sensor data"""
        
        # Update nodes with new sensor data
        for _, row in time_step_data.iterrows():
            sensor_id = row['sensor_id']
            if sensor_id in self.nodes:
                node = self.nodes[sensor_id]
                node.update_from_sensor_data(
                    row['temp_C'],
                    row['hpa_div_4'],
                    row['batterylevel'],
                    row['sensor_cycle']
                )
                
        # Simulate communication
        active_nodes = self.get_active_nodes()
        for node in active_nodes:
            # Send data packet
            if node.send_packet():
                # CH receives and processes
                if node.parent_ch and node.parent_ch in self.nodes:
                    ch_node = self.nodes[node.parent_ch]
                    ch_node.receive_packet()
                    
            # Idle processing
            node.idle_cycle()
            
        # Periodic CH selection
        if self.time_step % self.ch_rotation_interval == 0:
            self.select_cluster_heads()
            
    def collect_metrics(self):
        """Collect metrics for current state"""
        active_nodes = self.get_active_nodes()
        
        # Basic communication metrics
        total_packets_sent = sum(node.packets_sent for node in self.nodes.values())
        total_packets_received = sum(node.packets_received for node in self.nodes.values())
        total_packets_dropped = sum(node.packets_dropped for node in self.nodes.values())
        
        # Calculate metrics
        pdr = total_packets_received / max(1, total_packets_sent)
        packet_drop_rate = total_packets_dropped / max(1, total_packets_sent)
        
        # Energy metrics
        total_energy_consumed = sum(
            node.initial_battery - node.current_battery 
            for node in self.nodes.values()
        )
        avg_energy_consumption = total_energy_consumed / len(self.nodes)
        
        # Load efficiency
        loads = [node.current_load for node in active_nodes]
        if loads:
            load_variance = np.var(loads)
            load_mean = np.mean(loads)
            load_efficiency = 1.0 / (1.0 + load_variance / max(0.01, load_mean))
        else:
            load_efficiency = 0
            
        # Network lifetime
        network_lifetime = len(active_nodes) / len(self.nodes)
        
        # Store metrics
        self.metrics['packet_drop'].append(packet_drop_rate)
        self.metrics['pdr'].append(pdr)
        self.metrics['energy_consumption'].append(avg_energy_consumption)
        self.metrics['load_efficiency'].append(load_efficiency)
        self.metrics['alive_nodes'].append(len(active_nodes))
        self.metrics['network_lifetime'].append(network_lifetime)
        self.metrics['throughput'].append(total_packets_received)
        self.metrics['simulation_time'].append(self.time_step)
        
    def run_simulation(self):
        """Run the complete simulation using dataset"""
        print("Starting EARLB dataset-based simulation...")
        
        # Group data by time steps
        time_groups = self.sensor_data.groupby('datetime')
        
        print(f"Processing {len(time_groups)} time steps...")
        
        for timestamp, group_data in time_groups:
            self.time_step += 1
            
            # Filter data for nodes we're tracking
            relevant_data = group_data[group_data['sensor_id'].isin(self.nodes.keys())]
            
            if len(relevant_data) > 0:
                # Process this time step
                self.process_time_step(relevant_data)
                
                # Collect metrics every 10 time steps
                if self.time_step % 10 == 0:
                    self.collect_metrics()
                    
                # Progress indicator
                if self.time_step % 100 == 0:
                    active_count = len(self.get_active_nodes())
                    print(f"Time step {self.time_step}: {active_count}/{len(self.nodes)} nodes active")
                    
            # Stop if no nodes are active
            if not self.get_active_nodes():
                print("All nodes have died. Simulation ended.")
                break
                
        print("Simulation completed!")
        return self.metrics
        
    def generate_report(self):
        """Generate comprehensive simulation report"""
        if not self.metrics['simulation_time']:
            print("No metrics collected!")
            return None
            
        report = {
            'simulation_parameters': {
                'dataset_path': self.dataset_path,
                'num_nodes': len(self.nodes),
                'total_time_steps': self.time_step,
                'ch_rotation_interval': self.ch_rotation_interval,
                'dataset_records': len(self.sensor_data)
            },
            'final_metrics': {},
            'performance_summary': {}
        }
        
        # Calculate final metrics
        for metric_name, values in self.metrics.items():
            if values and metric_name != 'simulation_time':
                report['final_metrics'][metric_name] = {
                    'final': values[-1],
                    'average': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Performance summary
        final_alive = self.metrics['alive_nodes'][-1] if self.metrics['alive_nodes'] else 0
        avg_pdr = np.mean(self.metrics['pdr']) if self.metrics['pdr'] else 0
        avg_load_eff = np.mean(self.metrics['load_efficiency']) if self.metrics['load_efficiency'] else 0
        
        report['performance_summary'] = {
            'network_lifetime_percentage': (final_alive / len(self.nodes)) * 100,
            'average_pdr': avg_pdr,
            'average_load_efficiency': avg_load_eff,
            'total_energy_consumed': sum(
                node.initial_battery - node.current_battery 
                for node in self.nodes.values()
            ),
            'node_details': {
                str(node.sensor_id): {
                    'sensor_type': node.sensor_type,
                    'initial_battery': node.initial_battery,
                    'final_battery': node.current_battery,
                    'packets_sent': node.packets_sent,
                    'packets_received': node.packets_received,
                    'is_active': node.is_active
                }
                for node in self.nodes.values()
            }
        }
        
        return report
        
    def plot_results(self):
        """Plot simulation results"""
        if not self.metrics['simulation_time']:
            print("No data to plot!")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('EARLB Dataset-Based Simulation Results', fontsize=14, fontweight='bold')
        
        times = self.metrics['simulation_time']
        
        # Plot metrics
        metrics_to_plot = [
            ('packet_drop', 'Packet Drop Rate', 'red'),
            ('pdr', 'Packet Delivery Ratio', 'blue'),
            ('energy_consumption', 'Energy Consumption (%)', 'green'),
            ('load_efficiency', 'Load Efficiency', 'orange'),
            ('alive_nodes', 'Alive Nodes', 'purple'),
            ('network_lifetime', 'Network Lifetime', 'brown')
        ]
        
        for i, (metric, title, color) in enumerate(metrics_to_plot):
            row, col = i // 3, i % 3
            if metric in self.metrics and self.metrics[metric]:
                axes[row, col].plot(times, self.metrics[metric], color=color, linewidth=2)
                axes[row, col].set_title(title)
                axes[row, col].set_xlabel('Time Steps')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'earlb_dataset_results_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Results plot saved as {filename}")
        plt.show()

def compare_datasets(dataset_paths):
    """Compare multiple datasets using EARLB algorithm"""
    print("="*60)
    print("EARLB Algorithm - Multi-Dataset Comparison")
    print("="*60)
    
    results = {}
    
    for i, dataset_path in enumerate(dataset_paths):
        print(f"\nProcessing Dataset {i+1}: {dataset_path}")
        print("-" * 50)
        
        # Check if file exists
        if not os.path.exists(dataset_path):
            print(f"Error: File '{dataset_path}' not found! Skipping...")
            continue
            
        try:
            # Run simulation for this dataset
            sim = EARLBDatasetSimulation(dataset_path)
            metrics = sim.run_simulation()
            report = sim.generate_report()
            
            if report:
                results[f"Dataset_{i+1}"] = {
                    'path': dataset_path,
                    'metrics': metrics,
                    'report': report
                }
                
                # Display summary
                ps = report['performance_summary']
                print(f"Results for {dataset_path}:")
                print(f"  Network Lifetime: {ps['network_lifetime_percentage']:.1f}%")
                print(f"  Average PDR: {ps['average_pdr']:.3f}")
                print(f"  Average Load Efficiency: {ps['average_load_efficiency']:.3f}")
                print(f"  Total Energy Consumed: {ps['total_energy_consumed']:.2f}%")
                
        except Exception as e:
            print(f"Error processing {dataset_path}: {e}")
            continue
    
    # Generate comparison report
    if len(results) > 1:
        generate_comparison_report(results)
    
    return results

def generate_comparison_report(results):
    """Generate comparison report for multiple datasets"""
    print("\n" + "="*60)
    print("DATASET COMPARISON SUMMARY")
    print("="*60)
    
    comparison_data = []
    
    for dataset_name, data in results.items():
        ps = data['report']['performance_summary']
        comparison_data.append({
            'Dataset': dataset_name,
            'Network_Lifetime_%': ps['network_lifetime_percentage'],
            'Average_PDR': ps['average_pdr'],
            'Load_Efficiency': ps['average_load_efficiency'],
            'Energy_Consumed_%': ps['total_energy_consumed']
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    print("\nPerformance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Save comparison report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f'earlb_comparison_report_{timestamp}.json'
    
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nComparison report saved as {comparison_file}")
    
    # Plot comparison
    plot_comparison(comparison_df)

def plot_comparison(comparison_df):
    """Plot comparison charts"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('EARLB Multi-Dataset Comparison', fontsize=14, fontweight='bold')
    
    metrics = ['Network_Lifetime_%', 'Average_PDR', 'Load_Efficiency', 'Energy_Consumed_%']
    titles = ['Network Lifetime (%)', 'Average PDR', 'Load Efficiency', 'Energy Consumed (%)']
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        row, col = i // 2, i % 2
        axes[row, col].bar(comparison_df['Dataset'], comparison_df[metric], color=color, alpha=0.7)
        axes[row, col].set_title(title)
        axes[row, col].set_xlabel('Dataset')
        axes[row, col].tick_params(axis='x', rotation=45)
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'earlb_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved as {filename}")
    plt.show()

def main():
    """Main function"""
    print("="*60)
    print("EARLB Algorithm - Dataset-Based Implementation")
    print("Enhanced Adaptive Rotation Load Balancing for WSN")
    print("="*60)
    
    # OPTION 1: Single Dataset Analysis
    # Simply change this path to your dataset file
    dataset_path = "sensor_data.csv"  # Change this to your actual file path
    
    # OPTION 2: Multiple Dataset Comparison
    # Uncomment the lines below to compare multiple datasets
    # dataset_paths = [
    #     "dataset1.csv",
    #     "dataset2.csv",
    #     "dataset3.csv"
    # ]
    # results = compare_datasets(dataset_paths)
    # return results
    
    # OPTION 3: File Dialog (uncomment if you want to browse for file)
    # try:
    #     import tkinter as tk
    #     from tkinter import filedialog
    #     root = tk.Tk()
    #     root.withdraw()  # Hide the main window
    #     dataset_path = filedialog.askopenfilename(
    #         title="Select your sensor dataset CSV file",
    #         filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    #     )
    #     if not dataset_path:
    #         print("No file selected. Exiting.")
    #         sys.exit(1)
    # except ImportError:
    #     print("tkinter not available. Using hardcoded path.")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file '{dataset_path}' not found!")
        print("Please update the dataset_path variable in the main() function")
        print("Expected CSV format with columns:")
        print("sort_id, date_d_m_y, time, sensor_id, sensor_type, temp_C, hpa_div_4, batterylevel, sensor_cycle")
        sys.exit(1)
    
    # Create and run simulation
    sim = EARLBDatasetSimulation(dataset_path)
    metrics = sim.run_simulation()
    
    # Generate report
    report = sim.generate_report()
    
    if report:
        # Display results
        print("\n" + "="*50)
        print("SIMULATION RESULTS")
        print("="*50)
        
        ps = report['performance_summary']
        print(f"Network Lifetime: {ps['network_lifetime_percentage']:.1f}%")
        print(f"Average PDR: {ps['average_pdr']:.3f}")
        print(f"Average Load Efficiency: {ps['average_load_efficiency']:.3f}")
        print(f"Total Energy Consumed: {ps['total_energy_consumed']:.2f}%")
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f'earlb_dataset_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Detailed report saved as {report_file}")
        
        # Plot results
        sim.plot_results()
    
    return metrics, report

if __name__ == "__main__":
    metrics, report = main()