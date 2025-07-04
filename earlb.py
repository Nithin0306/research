#!/usr/bin/env python3
"""
Enhanced Adaptive Rotation Load Balancing (EARLB) Algorithm
NS-3 Python Implementation for Wireless Sensor Networks
"""
import ns3
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class EARLBApplication(ns3.Application):
    """EARLB Application for NS-3 nodes"""
    
    def __init__(self):
        super(EARLBApplication, self).__init__()
        self.node_id = 0
        self.initial_energy = 2.0
        self.current_energy = 2.0
        self.is_cluster_head = False
        self.cluster_members = []
        self.parent_ch = None
        
        # Load balancing attributes
        self.current_load = 0.0
        self.processing_capacity = np.random.uniform(0.8, 1.5)
        self.communication_cost = np.random.uniform(0.1, 0.3)
        
        # Metrics
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_dropped = 0
        self.total_delay = 0.0
        self.response_times = []
        self.load_history = []
        
        # Energy consumption rates
        self.tx_power = 0.01    # Watts
        self.rx_power = 0.005   # Watts
        self.idle_power = 0.001 # Watts
        self.ch_extra_power = 0.02  # Additional CH power
        
        # Socket for communication
        self.socket = None
        self.port = 9999
        
    def SetNodeId(self, node_id):
        self.node_id = node_id
        
    def StartApplication(self):
        """Start the EARLB application"""
        # Create UDP socket
        tid = ns3.TypeId.LookupByName("ns3::UdpSocketFactory")
        self.socket = ns3.Socket.CreateSocket(self.GetNode(), tid)
        
        # Bind to address
        local_address = ns3.InetSocketAddress(ns3.Ipv4Address.GetAny(), self.port)
        self.socket.Bind(local_address)
        self.socket.SetRecvCallback(self.ReceivePacket)
        
        # Schedule periodic tasks
        self.ScheduleTransmit(ns3.Seconds(1.0))
        self.ScheduleEnergyUpdate(ns3.Seconds(0.1))
        
    def StopApplication(self):
        """Stop the application"""
        if self.socket:
            self.socket.Close()
            
    def ScheduleTransmit(self, dt):
        """Schedule packet transmission"""
        self.transmit_event = ns3.Simulator.Schedule(dt, self.SendPacket)
        
    def ScheduleEnergyUpdate(self, dt):
        """Schedule energy update"""
        self.energy_event = ns3.Simulator.Schedule(dt, self.UpdateEnergy)
        
    def SendPacket(self):
        """Send data packet"""
        if self.current_energy <= 0:
            return
            
        # Create packet
        packet_size = np.random.randint(64, 512)
        packet = ns3.Packet(packet_size)
        
        # Add EARLB header with node info
        header_data = {
            'node_id': self.node_id,
            'energy': self.current_energy,
            'load': self.current_load,
            'is_ch': self.is_cluster_head,
            'timestamp': ns3.Simulator.Now().GetSeconds()
        }
        
        # Determine destination
        if self.parent_ch is not None:
            # Send to cluster head
            dest_address = ns3.InetSocketAddress(
                self.GetDestinationAddress(self.parent_ch), self.port)
        else:
            # Send to base station (simulate with broadcast)
            dest_address = ns3.InetSocketAddress(
                ns3.Ipv4Address("255.255.255.255"), self.port)
        
        # Send packet
        self.socket.SendTo(packet, 0, dest_address)
        self.packets_sent += 1
        
        # Energy consumption for transmission
        tx_energy = self.tx_power * (packet_size / 1000.0) * 0.001  # Convert to Joules
        self.ConsumeEnergy(tx_energy)
        
        # Add load
        self.current_load += packet_size / 1000.0
        self.load_history.append(self.current_load)
        
        # Schedule next transmission
        next_interval = ns3.Seconds(np.random.exponential(2.0))
        self.ScheduleTransmit(next_interval)
        
    def ReceivePacket(self, socket):
        """Receive and process packet"""
        packet = socket.Recv()
        if packet and self.current_energy > 0:
            # Energy consumption for reception
            rx_energy = self.rx_power * (packet.GetSize() / 1000.0) * 0.001
            
            if self.is_cluster_head:
                # Additional processing energy for cluster heads
                rx_energy += self.ch_extra_power * (packet.GetSize() / 1000.0) * 0.001
                
            self.ConsumeEnergy(rx_energy)
            self.packets_received += 1
            
            # Process load
            self.current_load += packet.GetSize() / 2000.0  # CH processes more efficiently
            
    def ConsumeEnergy(self, amount):
        """Consume energy and check if node dies"""
        self.current_energy = max(0, self.current_energy - amount)
        
    def UpdateEnergy(self):
        """Update energy consumption for idle state"""
        if self.current_energy > 0:
            # Idle energy consumption
            idle_consumption = self.idle_power * 0.1 * 0.001  # 0.1 second interval
            self.ConsumeEnergy(idle_consumption)
            
            # Process current load
            processed = min(self.current_load, self.processing_capacity * 0.1)
            self.current_load = max(0, self.current_load - processed)
            
            # Schedule next energy update
            self.ScheduleEnergyUpdate(ns3.Seconds(0.1))
            
    def GetDestinationAddress(self, dest_node_id):
        """Get destination IP address for a node"""
        # Simplified addressing scheme
        return ns3.Ipv4Address(f"10.1.1.{dest_node_id + 1}")
        
    def GetEnergyRatio(self):
        """Get remaining energy ratio"""
        return self.current_energy / self.initial_energy if self.initial_energy > 0 else 0
        
    def CalculateCHProbability(self):
        """Calculate cluster head selection probability"""
        if self.current_energy <= 0.1:
            return 0.0
            
        energy_factor = self.GetEnergyRatio()
        load_factor = max(0.1, 1.0 - (self.current_load / 10.0))
        comm_factor = 1.0 - self.communication_cost
        
        return (energy_factor * 0.5 + load_factor * 0.3 + comm_factor * 0.2)

class EARLBSimulation:
    """Main EARLB simulation class for NS-3"""
    
    def __init__(self, num_nodes=50):
        self.num_nodes = num_nodes
        self.nodes = ns3.NodeContainer()
        self.devices = None
        self.interfaces = None
        self.applications = []
        
        # Metrics storage
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
        
        self.current_chs = set()
        self.ch_rotation_interval = 10.0  # seconds
        
    def SetupTopology(self):
        """Setup network topology"""
        print(f"Setting up topology with {self.num_nodes} nodes...")
        
        # Create nodes
        self.nodes.Create(self.num_nodes)
        
        # Set mobility model
        mobility = ns3.MobilityHelper()
        position_alloc = ns3.ListPositionAllocator()
        
        # Random positions in 100x100 area
        for i in range(self.num_nodes):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            position_alloc.Add(ns3.Vector(x, y, 0))
            
        mobility.SetPositionAllocator(position_alloc)
        mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
        mobility.Install(self.nodes)
        
    def SetupWifiNetwork(self):
        """Setup WiFi network for WSN simulation"""
        # WiFi configuration
        wifi = ns3.WifiHelper()
        wifi.SetStandard(ns3.WIFI_STANDARD_80211b)
        
        # MAC and PHY configuration
        mac = ns3.WifiMacHelper()
        mac.SetType("ns3::AdhocWifiMac")
        
        phy = ns3.YansWifiPhyHelper()
        channel = ns3.YansWifiChannelHelper.Default()
        phy.SetChannel(channel.Create())
        
        # Install WiFi
        self.devices = wifi.Install(phy, mac, self.nodes)
        
        # IP configuration
        stack = ns3.InternetStackHelper()
        stack.Install(self.nodes)
        
        address = ns3.Ipv4AddressHelper()
        address.SetBase("10.1.1.0", "255.255.255.0")
        self.interfaces = address.Assign(self.devices)
        
    def InstallApplications(self):
        """Install EARLB applications on nodes"""
        print("Installing EARLB applications...")
        
        for i in range(self.num_nodes):
            app = EARLBApplication()
            app.SetNodeId(i)
            self.nodes.Get(i).AddApplication(app)
            self.applications.append(app)
            
            # Start applications at random times
            start_time = ns3.Seconds(np.random.uniform(0.1, 1.0))
            app.SetStartTime(start_time)
            app.SetStopTime(ns3.Seconds(200.0))
            
    def ScheduleCHSelection(self):
        """Schedule cluster head selection"""
        ns3.Simulator.Schedule(ns3.Seconds(1.0), self.SelectClusterHeads)
        
    def SelectClusterHeads(self):
        """EARLB cluster head selection algorithm"""
        alive_nodes = [app for app in self.applications if app.current_energy > 0]
        
        if not alive_nodes:
            return
            
        # Calculate optimal number of CHs
        optimal_ch_count = max(1, int(len(alive_nodes) * 0.07))
        
        # Calculate probabilities
        candidates = []
        for app in alive_nodes:
            prob = app.CalculateCHProbability()
            if prob > 0.1:
                candidates.append((app, prob))
                
        # Select CHs
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected_chs = [app for app, _ in candidates[:optimal_ch_count]]
        
        # Reset CH status
        for app in self.applications:
            app.is_cluster_head = False
            app.cluster_members = []
            
        # Set new CHs
        new_ch_set = set()
        for ch_app in selected_chs:
            ch_app.is_cluster_head = True
            new_ch_set.add(ch_app.node_id)
            
        # Form clusters
        self.FormClusters(selected_chs, alive_nodes)
        
        # Update CH tracking
        self.current_chs = new_ch_set
        
        # Schedule next CH selection
        next_selection = ns3.Seconds(self.ch_rotation_interval)
        ns3.Simulator.Schedule(next_selection, self.SelectClusterHeads)
        
    def FormClusters(self, cluster_heads, regular_nodes):
        """Form clusters around selected cluster heads"""
        non_ch_nodes = [app for app in regular_nodes if not app.is_cluster_head]
        
        for node_app in non_ch_nodes:
            best_ch = None
            best_score = float('inf')
            
            for ch_app in cluster_heads:
                # Simplified distance and load calculation
                distance = np.random.uniform(10, 100)
                energy_factor = ch_app.GetEnergyRatio()
                load_factor = max(0.1, 1.0 - (ch_app.current_load / 20.0))
                
                score = distance / (energy_factor * load_factor)
                
                if score < best_score:
                    best_score = score
                    best_ch = ch_app
                    
            if best_ch:
                node_app.parent_ch = best_ch.node_id
                best_ch.cluster_members.append(node_app.node_id)
                
    def ScheduleMetricsCollection(self):
        """Schedule periodic metrics collection"""
        ns3.Simulator.Schedule(ns3.Seconds(2.0), self.CollectMetrics)
        
    def CollectMetrics(self):
        """Collect and store simulation metrics"""
        current_time = ns3.Simulator.Now().GetSeconds()
        alive_nodes = [app for app in self.applications if app.current_energy > 0]
        
        # Basic metrics
        total_packets_sent = sum(app.packets_sent for app in self.applications)
        total_packets_received = sum(app.packets_received for app in self.applications)
        total_packets_dropped = sum(app.packets_dropped for app in self.applications)
        
        # Calculate metrics
        pdr = total_packets_received / max(1, total_packets_sent)
        packet_drop_rate = total_packets_dropped / max(1, total_packets_sent)
        
        # Energy metrics
        total_energy_consumed = sum(app.initial_energy - app.current_energy 
                                  for app in self.applications)
        avg_energy_consumption = total_energy_consumed / self.num_nodes
        
        # Load efficiency
        loads = [app.current_load for app in alive_nodes]
        if loads:
            load_variance = np.var(loads)
            load_mean = np.mean(loads)
            load_efficiency = 1.0 / (1.0 + load_variance / max(0.01, load_mean))
        else:
            load_efficiency = 0
            
        # Store metrics
        self.metrics['packet_drop'].append(packet_drop_rate)
        self.metrics['pdr'].append(pdr)
        self.metrics['energy_consumption'].append(avg_energy_consumption)
        self.metrics['load_efficiency'].append(load_efficiency)
        self.metrics['alive_nodes'].append(len(alive_nodes))
        self.metrics['network_lifetime'].append(len(alive_nodes) / self.num_nodes)
        self.metrics['throughput'].append(total_packets_received)
        self.metrics['simulation_time'].append(current_time)
        
        # Schedule next collection
        if current_time < 195.0:  # Stop before simulation ends
            ns3.Simulator.Schedule(ns3.Seconds(2.0), self.CollectMetrics)
            
    def RunSimulation(self):
        """Run the complete NS-3 simulation"""
        print("Starting EARLB NS-3 simulation...")
        
        # Setup network
        self.SetupTopology()
        self.SetupWifiNetwork()
        self.InstallApplications()
        
        # Schedule periodic tasks
        self.ScheduleCHSelection()
        self.ScheduleMetricsCollection()
        
        # Enable packet capture (optional)
        # ns3.PointToPointHelper.EnablePcapAll("earlb-simulation")
        
        # Run simulation
        ns3.Simulator.Stop(ns3.Seconds(200.0))
        ns3.Simulator.Run()
        ns3.Simulator.Destroy()
        
        print("Simulation completed!")
        return self.metrics
        
    def GenerateReport(self):
        """Generate comprehensive simulation report"""
        if not self.metrics['simulation_time']:
            print("No metrics collected!")
            return None
            
        report = {
            'simulation_parameters': {
                'num_nodes': self.num_nodes,
                'simulation_duration': max(self.metrics['simulation_time']),
                'ch_rotation_interval': self.ch_rotation_interval
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
            'network_lifetime_percentage': (final_alive / self.num_nodes) * 100,
            'average_pdr': avg_pdr,
            'average_load_efficiency': avg_load_eff,
            'total_energy_consumed': sum(app.initial_energy - app.current_energy 
                                       for app in self.applications)
        }
        
        return report
        
    def PlotResults(self):
        """Plot simulation results"""
        if not self.metrics['simulation_time']:
            print("No data to plot!")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('EARLB NS-3 Simulation Results', fontsize=14, fontweight='bold')
        
        times = self.metrics['simulation_time']
        
        # Plot metrics
        metrics_to_plot = [
            ('packet_drop', 'Packet Drop Rate', 'red'),
            ('pdr', 'Packet Delivery Ratio', 'blue'),
            ('energy_consumption', 'Energy Consumption (J)', 'green'),
            ('load_efficiency', 'Load Efficiency', 'orange'),
            ('alive_nodes', 'Alive Nodes', 'purple'),
            ('network_lifetime', 'Network Lifetime', 'brown')
        ]
        
        for i, (metric, title, color) in enumerate(metrics_to_plot):
            row, col = i // 3, i % 3
            if metric in self.metrics and self.metrics[metric]:
                axes[row, col].plot(times, self.metrics[metric], color=color, linewidth=2)
                axes[row, col].set_title(title)
                axes[row, col].set_xlabel('Time (seconds)')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'earlb_ns3_results_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Results plot saved as {filename}")
        plt.show()

def main():
    """Main function"""
    print("="*60)
    print("EARLB Algorithm - NS-3 Python Implementation")
    print("Enhanced Adaptive Rotation Load Balancing for WSN")
    print("="*60)
    
    # Create and run simulation
    sim = EARLBSimulation(num_nodes=50)
    metrics = sim.RunSimulation()
    
    # Generate report
    report = sim.GenerateReport()
    
    if report:
        # Display results
        print("\n" + "="*50)
        print("SIMULATION RESULTS")
        print("="*50)
        
        ps = report['performance_summary']
        print(f"Network Lifetime: {ps['network_lifetime_percentage']:.1f}%")
        print(f"Average PDR: {ps['average_pdr']:.3f}")
        print(f"Average Load Efficiency: {ps['average_load_efficiency']:.3f}")
        print(f"Total Energy Consumed: {ps['total_energy_consumed']:.2f} J")
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f'earlb_ns3_report_{timestamp}.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Detailed report saved as {report_file}")
        
        # Plot results
        sim.PlotResults()
    
    return metrics, report

if __name__ == "__main__":
    # Check if NS-3 is available
    try:
        import ns3
        metrics, report = main()
    except ImportError:
        print("Error: NS-3 Python bindings not found!")
        print("Please ensure NS-3 is properly installed with Python bindings enabled.")
        print("Installation instructions:")
        print("1. Install NS-3: https://www.nsnam.org/wiki/Installation")
        print("2. Enable Python bindings during configuration")
        print("3. Set PYTHONPATH to include NS-3 bindings")
        sys.exit(1)
