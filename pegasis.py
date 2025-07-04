import ns.applications
import ns.core
import ns.internet
import ns.mobility
import ns.network

import math
import random

NUM_NODES = 10
FIELD_SIZE = 100.0
BASE_STATION = (50.0, 150.0)

class NodeInfo:
    def __init__(self, node, position, node_id):
        self.node = node
        self.position = position
        self.id = node_id
        self.energy = 100.0
        self.next_node = None

    def distance(self, other):
        dx = self.position.x - other.position.x
        dy = self.position.y - other.position.y
        return math.hypot(dx, dy)

def create_nodes():
    nodes = ns.network.NodeContainer()
    nodes.Create(NUM_NODES)
    return nodes

def assign_positions(nodes):
    mobility = ns.mobility.MobilityHelper()
    position_alloc = ns.mobility.ListPositionAllocator()

    node_infos = []

    # Assign random positions
    for i in range(NUM_NODES):
        x = random.uniform(0, FIELD_SIZE)
        y = random.uniform(0, FIELD_SIZE)
        position_alloc.Add(ns.core.Vector(x, y, 0.0))

    mobility.SetPositionAllocator(position_alloc)
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
    mobility.Install(nodes)

    # Correct way to get position from mobility model
    for i in range(NUM_NODES):
        node = nodes.Get(i)
        mobility_model = node.GetObject(ns.mobility.MobilityModel.GetTypeId())
        position = mobility_model.GetPosition()
        node_infos.append(NodeInfo(node, position, i))

    return node_infos

def form_chain(node_infos):
    unvisited = node_infos[:]
    chain = []
    current = unvisited.pop(0)
    chain.append(current)
    while unvisited:
        nearest = min(unvisited, key=lambda n: current.distance(n))
        current.next_node = nearest
        chain.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    return chain

def run_pegasis_round(chain, round_num):
    leader_index = round_num % len(chain)
    leader = chain[leader_index]
    print(f"\nRound {round_num + 1} - Leader: Node {leader.id}")

    for i, node in enumerate(chain):
        if i != leader_index:
            receiver = chain[i + 1] if i < leader_index else chain[i - 1]
            dist = node.distance(receiver)
            node.energy -= 0.5 * dist  # Sending energy
            receiver.energy -= 0.2 * dist  # Receiving energy

    base = ns.core.Vector(BASE_STATION[0], BASE_STATION[1], 0.0)
    dist_to_bs = math.hypot(leader.position.x - base.x, leader.position.y - base.y)
    leader.energy -= 1.0 * dist_to_bs  # Leader sends to base station

    for node in chain:
        print(f"Node {node.id}: Energy = {node.energy:.2f}")

def simulate_pegasis(rounds=5):
    nodes = create_nodes()
    node_infos = assign_positions(nodes)
    chain = form_chain(node_infos)

    print("Initial Chain:")
    for node in chain:
        print(f"Node {node.id} -> ", end="")
    print("END")

    for round_num in range(rounds):
        run_pegasis_round(chain, round_num)

simulate_pegasis(rounds=5)
