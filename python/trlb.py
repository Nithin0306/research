import csv
import math
import os
from collections import defaultdict

class EARLBDatasetAnalyzer:
    def __init__(self, dataset_path="sensor_data.csv"):
        self.dataset_path = dataset_path
        self.dataset = []
        self.nodes_data = defaultdict(list)
        self.performance_metrics = {
            'energy_consumption': [],
            'packet_delivery_ratio': [],
            'throughput': [],
            'latency': [],
            'cluster_head_rotations': 0,
            'network_lifetime': 0,
            'load_balancing_efficiency': []
        }
        self.cluster_heads = []
        self.current_cycle = 0

    def load_dataset(self):
        try:
            if not os.path.exists(self.dataset_path):
                print(f"Dataset file {self.dataset_path} not found. Using simulated data.")
                return self._generate_simulated_data()

            with open(self.dataset_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                required_columns = ['sort_id', 'date_d_m_y', 'time', 'sensor_id',
                                    'sensor_type', 'temp_C', 'hpa_div_4', 'batterylevel', 'sensor_cycle']

                if not all(col in reader.fieldnames for col in required_columns):
                    print("Missing required columns.")
                    return self._generate_simulated_data()

                for row in reader:
                    try:
                        processed_row = {
                            'sort_id': int(row['sort_id']),
                            'date_d_m_y': row['date_d_m_y'],
                            'time': row['time'],
                            'sensor_id': int(row['sensor_id']),
                            'sensor_type': row['sensor_type'],
                            'temp_C': float(row['temp_C']),
                            'hpa_div_4': float(row['hpa_div_4']),
                            'batterylevel': float(row['batterylevel']),
                            'sensor_cycle': int(row['sensor_cycle'])
                        }
                        self.dataset.append(processed_row)
                        sensor_id = processed_row['sensor_id']
                        if sensor_id <= 50:
                            self.nodes_data[sensor_id].append(processed_row)
                    except Exception:
                        continue

            for sensor_id in self.nodes_data:
                self.nodes_data[sensor_id].sort(key=lambda x: x['sensor_cycle'])

            print(f"Dataset loaded: {len(self.dataset)} records")
            return True

        except Exception as e:
            print(f"Error: {e}")
            return self._generate_simulated_data()

    def _generate_simulated_data(self):
        print("Generating simulated data...")
        for sensor_id in range(1, 51):
            for cycle in range(10):
                data = {
                    'sort_id': sensor_id * 10 + cycle,
                    'date_d_m_y': '01/01/2024',
                    'time': f'{cycle}:00:00',
                    'sensor_id': sensor_id,
                    'sensor_type': 'temperature',
                    'temp_C': 20.0 + (sensor_id % 10) + (cycle * 0.5),
                    'hpa_div_4': 1000.0 + (sensor_id % 50),
                    'batterylevel': max(10.0, 100.0 - (cycle * 8) - (sensor_id % 20)),
                    'sensor_cycle': cycle
                }
                self.dataset.append(data)
                self.nodes_data[sensor_id].append(data)
        return True

    def get_node_energy(self, node_id, cycle):
        if node_id not in self.nodes_data:
            return 50.0
        for data in reversed(self.nodes_data[node_id]):
            if data['sensor_cycle'] <= cycle:
                return max(0.0, data['batterylevel'])
        return 100.0

    def get_node_load(self, node_id, cycle):
        if node_id not in self.nodes_data:
            return 0.5
        recent_data = [d for d in self.nodes_data[node_id] if d['sensor_cycle'] <= cycle][-5:]
        if not recent_data:
            return 0.5
        temps = [d['temp_C'] for d in recent_data]
        mean = sum(temps) / len(temps)
        variance = sum((t - mean) ** 2 for t in temps) / len(temps)
        return min(1.0, variance / 10.0 + 0.1)

    def calculate_earlb_fitness(self, node_id, cycle):
        energy = self.get_node_energy(node_id, cycle)
        load = self.get_node_load(node_id, cycle)
        return (energy / 100.0) * 0.6 + (1.0 - load) * 0.4

    def select_cluster_heads_earlb(self, cycle, num_clusters=5):
        fitness_scores = {
            node_id: self.calculate_earlb_fitness(node_id, cycle)
            for node_id in range(1, 51)
            if self.get_node_energy(node_id, cycle) > 10.0
        }
        sorted_nodes = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:num_clusters]]

    def calculate_load_balancing_efficiency(self, cluster_heads, cycle):
        loads = [self.get_node_load(ch, cycle) for ch in cluster_heads]
        if not loads:
            return 0.0
        mean = sum(loads) / len(loads)
        variance = sum((l - mean) ** 2 for l in loads) / len(loads)
        std_dev = math.sqrt(variance)
        return max(0.0, 100.0 - (std_dev * 100.0))

    def simulate_earlb_clustering(self, cycle):
        cluster_heads = self.select_cluster_heads_earlb(cycle)
        self.cluster_heads = cluster_heads

        total_energy = 0.0
        total_load = 0.0
        active_nodes = 0

        for node_id in range(1, 51):
            energy = self.get_node_energy(node_id, cycle)
            load = self.get_node_load(node_id, cycle)
            if energy > 0:
                total_energy += energy
                total_load += load
                active_nodes += 1

        if active_nodes > 0:
            avg_energy = total_energy / active_nodes
            avg_load = total_load / active_nodes
            self.performance_metrics['energy_consumption'].append(100.0 - avg_energy)
            self.performance_metrics['packet_delivery_ratio'].append(
                min(100.0, 85.0 + (avg_energy * 0.1) + (10.0 - avg_load * 10.0)))
            self.performance_metrics['throughput'].append(
                (avg_energy / 100.0) * (1.0 - avg_load) * 10.0)
            self.performance_metrics['latency'].append(
                max(5.0, 50.0 - (avg_energy * 0.3) + (avg_load * 20.0)))
            self.performance_metrics['load_balancing_efficiency'].append(
                self.calculate_load_balancing_efficiency(cluster_heads, cycle))

        self.performance_metrics['cluster_head_rotations'] += 1
        return cluster_heads

def main():
    analyzer = EARLBDatasetAnalyzer()
    if not analyzer.load_dataset():
        print("Dataset failed to load.")
        return

    max_cycles = min(10, max(d['sensor_cycle'] for d in analyzer.dataset) + 1) if analyzer.dataset else 10

    print("\nSimulating EARLB clustering...\n")
    for cycle in range(max_cycles):
        print(f"Cycle {cycle}:")
        chs = analyzer.simulate_earlb_clustering(cycle)
        print(f"  Selected Cluster Heads: {chs}")

    print("\n" + "="*60)
    print("EARLB PERFORMANCE ANALYSIS RESULTS")
    print("="*60)
    print(f"Average Energy Consumption: {sum(analyzer.performance_metrics['energy_consumption'])/max_cycles:.2f}%")
    print(f"Average PDR: {sum(analyzer.performance_metrics['packet_delivery_ratio'])/max_cycles:.2f}%")
    print(f"Average Throughput: {sum(analyzer.performance_metrics['throughput'])/max_cycles:.2f} Mbps")
    print(f"Average Latency: {sum(analyzer.performance_metrics['latency'])/max_cycles:.2f} ms")
    print(f"Average Load Balancing Efficiency: {sum(analyzer.performance_metrics['load_balancing_efficiency'])/max_cycles:.2f}%")
    print(f"Total Cluster Head Rotations: {analyzer.performance_metrics['cluster_head_rotations']}")
    print(f"Network Lifetime (Cycles): {max_cycles}")
    print("\nSimulation completed successfully.")

if __name__ == "__main__":
    main()
