import ns.core
import ns.network
import ns.mobility
import ns.applications
import ns.internet

import math
import random
import csv
from collections import defaultdict

# Configuration
NUM_NODES = 50
FIELD_SIZE = 100.0
BASE_STATION = ns.core.Vector(50.0, 150.0, 0.0)
DATASET_FILE = "sensor_data.csv"

class NodeInfo:
    def __init__(self, node, position, node_id, sensor_type="default"):
        self.node = node
        self.position = position
        self.id = node_id
        self.sensor_type = sensor_type
        self.energy = 100.0
        self.initial_energy = 100.0
        self.next_node = None
        self.data_packets = []
        self.total_data_sent = 0
        self.alive = True
        
    def distance(self, other_position):
        dx = self.position.x - other_position.x
        dy = self.position.y - other_position.y
        return math.hypot(dx, dy)
    
    def add_data_packet(self, temp, pressure, battery_level, cycle):
        self.data_packets.append({
            'temp': temp,
            'pressure': pressure,
            'battery_level': battery_level,
            'cycle': cycle
        })
    
    def get_energy_consumption_factor(self):
        base_factor = 1.0
        if self.sensor_type == "temperature":
            base_factor = 0.8
        elif self.sensor_type == "pressure":
            base_factor = 1.2
        elif self.sensor_type == "humidity":
            base_factor = 0.9
        
        data_factor = 1.0 + (len(self.data_packets) * 0.1)
        return base_factor * data_factor

class DatasetLoader:
    def __init__(self, filename):
        self.filename = filename
        self.sensor_data = defaultdict(list)
        self.sensor_positions = {}
        self.sensor_types = {}
        
    def load_dataset(self):
        try:
            with open(self.filename, 'r') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    sensor_id = int(row['sensor_id'])
                    self.sensor_data[sensor_id].append({
                        'sort_id': int(row['sort_id']),
                        'temp_C': float(row['temp_C']),
                        'hpa_div_4': float(row['hpa_div_4']),
                        'battery_level': float(row['batterylevel']),
                        'sensor_cycle': int(row['sensor_cycle'])
                    })
                    
                    if sensor_id not in self.sensor_types:
                        self.sensor_types[sensor_id] = row['sensor_type']
                        
            print(f"Dataset loaded: {len(self.sensor_data)} sensors")
            return True
            
        except FileNotFoundError:
            print(f"Dataset file not found. Using simulated data.")
            return False
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def generate_positions_from_data(self):
        for sensor_id in self.sensor_data.keys():
            if sensor_id < NUM_NODES:
                data_points = len(self.sensor_data[sensor_id])
                x = (sensor_id * 7 + data_points * 3) % int(FIELD_SIZE)
                y = (sensor_id * 11 + data_points * 5) % int(FIELD_SIZE)
                self.sensor_positions[sensor_id] = ns.core.Vector(float(x), float(y), 0.0)

def create_nodes():
    nodes = ns.network.NodeContainer()
    nodes.Create(NUM_NODES)
    return nodes

def assign_positions_with_dataset(nodes, dataset_loader):
    mobility = ns.mobility.MobilityHelper()
    position_alloc = ns.mobility.ListPositionAllocator()
    
    node_infos = []
    
    dataset_loader.generate_positions_from_data()
    
    for i in range(NUM_NODES):
        if i in dataset_loader.sensor_positions:
            pos = dataset_loader.sensor_positions[i]
        else:
            pos = ns.core.Vector(
                random.uniform(0, FIELD_SIZE),
                random.uniform(0, FIELD_SIZE),
                0.0
            )
        position_alloc.Add(pos)
    
    mobility.SetPositionAllocator(position_alloc)
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(nodes)
    
    for i in range(NUM_NODES):
        node = nodes.Get(i)
        mobility_model = node.GetObject(ns.mobility.MobilityModel.GetTypeId())
        position = mobility_model.GetPosition()
        
        sensor_type = dataset_loader.sensor_types.get(i, "default")
        node_info = NodeInfo(node, position, i, sensor_type)
        
        if i in dataset_loader.sensor_data:
            for data_point in dataset_loader.sensor_data[i]:
                node_info.add_data_packet(
                    data_point['temp_C'],
                    data_point['hpa_div_4'],
                    data_point['battery_level'],
                    data_point['sensor_cycle']
                )
        
        node_infos.append(node_info)
    
    return node_infos

def form_chain(node_infos):
    unvisited = [node for node in node_infos if node.alive]
    chain = []
    
    if not unvisited:
        return chain
    
    current = unvisited.pop(0)
    chain.append(current)
    
    while unvisited:
        nearest = min(unvisited, key=lambda n: current.distance(n.position))
        current.next_node = nearest
        chain.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    return chain

def calculate_energy_consumption(node, distance, operation="send"):
    base_energy = 0.5 if operation == "send" else 0.2
    energy_factor = node.get_energy_consumption_factor()
    
    if distance < 10:
        distance_factor = 1.0
    elif distance < 50:
        distance_factor = 1.5
    else:
        distance_factor = 2.0
    
    return base_energy * distance * energy_factor * distance_factor

def run_pegasis_round(chain, round_num):
    if not chain:
        return
    
    leader_index = round_num % len(chain)
    leader = chain[leader_index]
    
    print(f"\nRound {round_num + 1} - Leader: Node {leader.id} (Type: {leader.sensor_type})")
    
    total_data_aggregated = 0
    
    for i, node in enumerate(chain):
        if not node.alive:
            continue
            
        if i != leader_index:
            if i < leader_index:
                receiver = chain[i + 1] if i + 1 < len(chain) else chain[i - 1]
            else:
                receiver = chain[i - 1] if i - 1 >= 0 else chain[i + 1]
            
            if receiver and receiver.alive:
                dist = node.distance(receiver.position)
                send_energy = calculate_energy_consumption(node, dist, "send")
                node.energy -= send_energy
                node.total_data_sent += len(node.data_packets)
                
                receive_energy = calculate_energy_consumption(receiver, dist, "receive")
                receiver.energy -= receive_energy
                
                total_data_aggregated += len(node.data_packets)
                
                if node.energy <= 0:
                    node.alive = False
                    print(f"Node {node.id} has died!")
    
    if leader.alive:
        dist_to_bs = leader.distance(BASE_STATION)
        bs_energy = calculate_energy_consumption(leader, dist_to_bs, "send") * 2
        leader.energy -= bs_energy
        leader.total_data_sent += total_data_aggregated
        
        if leader.energy <= 0:
            leader.alive = False
            print(f"Leader Node {leader.id} has died!")
    
    alive_nodes = sum(1 for node in chain if node.alive)
    print(f"Alive nodes: {alive_nodes}/{len(chain)}")
    
    for node in chain:
        status = "ALIVE" if node.alive else "DEAD"
        print(f"Node {node.id}: Energy = {node.energy:.2f}, Data Sent = {node.total_data_sent}, Status = {status}")

def calculate_performance_metrics(node_infos, total_rounds):
    alive_nodes = sum(1 for node in node_infos if node.alive)
    dead_nodes = len(node_infos) - alive_nodes
    
    total_energy_consumed = sum(node.initial_energy - node.energy for node in node_infos)
    total_data_transmitted = sum(node.total_data_sent for node in node_infos)
    
    avg_energy_remaining = sum(node.energy for node in node_infos if node.alive) / max(alive_nodes, 1)
    network_lifetime = total_rounds if alive_nodes > 0 else total_rounds
    
    print(f"\n{'='*60}")
    print("PERFORMANCE METRICS")
    print(f"{'='*60}")
    print(f"Total Nodes: {len(node_infos)}")
    print(f"Alive Nodes: {alive_nodes}")
    print(f"Dead Nodes: {dead_nodes}")
    print(f"Network Lifetime: {network_lifetime} rounds")
    print(f"Total Energy Consumed: {total_energy_consumed:.2f} J")
    print(f"Total Data Transmitted: {total_data_transmitted} packets")
    print(f"Average Energy Remaining: {avg_energy_remaining:.2f} J")
    print(f"Energy Efficiency: {total_data_transmitted/max(total_energy_consumed, 1):.2f} packets/J")
    print(f"Network Survival Rate: {(alive_nodes/len(node_infos))*100:.1f}%")
    
    sensor_type_stats = defaultdict(lambda: {'alive': 0, 'dead': 0, 'energy': 0})
    for node in node_infos:
        if node.alive:
            sensor_type_stats[node.sensor_type]['alive'] += 1
            sensor_type_stats[node.sensor_type]['energy'] += node.energy
        else:
            sensor_type_stats[node.sensor_type]['dead'] += 1
    
    print(f"\nSensor Type Performance:")
    for sensor_type, stats in sensor_type_stats.items():
        total = stats['alive'] + stats['dead']
        survival_rate = (stats['alive'] / total) * 100 if total > 0 else 0
        avg_energy = stats['energy'] / stats['alive'] if stats['alive'] > 0 else 0
        print(f"  {sensor_type}: {stats['alive']}/{total} alive ({survival_rate:.1f}%), Avg Energy: {avg_energy:.2f}J")

def simulate_pegasis_with_dataset(rounds=10):
    print("Starting PEGASIS Simulation with Dataset Integration")
    print(f"{'='*60}")
    
    dataset_loader = DatasetLoader(DATASET_FILE)
    dataset_loaded = dataset_loader.load_dataset()
    
    if not dataset_loaded:
        print("Proceeding with simulated sensor data...")
        for i in range(NUM_NODES):
            dataset_loader.sensor_types[i] = random.choice(["temperature", "pressure", "humidity"])
            for j in range(random.randint(5, 15)):
                if i not in dataset_loader.sensor_data:
                    dataset_loader.sensor_data[i] = []
                dataset_loader.sensor_data[i].append({
                    'sort_id': j,
                    'temp_C': random.uniform(20, 35),
                    'hpa_div_4': random.uniform(900, 1100),
                    'battery_level': random.uniform(70, 100),
                    'sensor_cycle': j
                })
    
    nodes = create_nodes()
    node_infos = assign_positions_with_dataset(nodes, dataset_loader)
    
    print(f"\nInitial Network Setup:")
    print(f"Number of nodes: {len(node_infos)}")
    print(f"Field size: {FIELD_SIZE}x{FIELD_SIZE}")
    print(f"Base station location: ({BASE_STATION.x}, {BASE_STATION.y})")
    
    for round_num in range(rounds):
        chain = form_chain(node_infos)
        
        if not chain:
            print(f"\nAll nodes are dead. Simulation ended at round {round_num + 1}")
            break
        
        if round_num == 0:
            print(f"\nInitial Chain Formation:")
            for i, node in enumerate(chain):
                next_id = chain[i+1].id if i+1 < len(chain) else "END"
                print(f"Node {node.id} -> {next_id}")
        
        run_pegasis_round(chain, round_num)
        
        alive_nodes = sum(1 for node in node_infos if node.alive)
        if alive_nodes == 0:
            print(f"\nNetwork completely depleted at round {round_num + 1}")
            break
    
    calculate_performance_metrics(node_infos, rounds)

if __name__ == "__main__":
    ns.core.CommandLine().Parse(sys.argv)
    simulate_pegasis_with_dataset(rounds=10)