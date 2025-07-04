import ns.core
import ns.network
import ns.internet
import ns.mobility
import ns.applications
import ns.energy
import ns.wifi

# Create 50 nodes
num_nodes = 50
nodes = ns.network.NodeContainer()
nodes.Create(num_nodes)

# Set mobility model (random or grid)
mobility = ns.mobility.MobilityHelper()
position_alloc = ns.mobility.ListPositionAllocator()
for i in range(num_nodes):
    x = (i % 10) * 20  # grid-like
    y = (i // 10) * 20
    position_alloc.Add(ns.core.Vector(x, y, 0))
mobility.SetPositionAllocator(position_alloc)
mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel")
mobility.Install(nodes)

# Wifi for communication
wifi_phy = ns.wifi.YansWifiPhyHelper.Default()
wifi_channel = ns.wifi.YansWifiChannelHelper.Default()
wifi_phy.SetChannel(wifi_channel.Create())

wifi_mac = ns.wifi.WifiMacHelper()
wifi_helper = ns.wifi.WifiHelper()
wifi_helper.SetRemoteStationManager("ns3::AarfWifiManager")
wifi_mac.SetType("ns3::StaWifiMac", "Ssid", ns.wifi.SsidValue(ns.wifi.Ssid("trlb-wsn")))

devices = wifi_helper.Install(wifi_phy, wifi_mac, nodes)

# Install Internet stack
stack = ns.internet.InternetStackHelper()
stack.Install(nodes)

address = ns.internet.Ipv4AddressHelper()
address.SetBase(ns.network.Ipv4Address("10.1.1.0"), ns.network.Ipv4Mask("255.255.255.0"))
interfaces = address.Assign(devices)

# TRLB Load Balancing Logic (simplified stub)
# -------------------------------------------
# You would implement logic here to:
# 1. Select cluster heads based on rotation and energy
# 2. Create tiers for CH selection
# 3. Re-assign clusters every round

def simulate_trlb_clustering():
    print("Running TRLB clustering logic...")
    # Cluster head selection, tier assignment, energy-aware rotation
    pass  # This needs actual logic per your paper

# Application Layer: Send dummy packets from each node to sink (node 0)
sink_address = interfaces.GetAddress(0)
port = 9
packet_sink_helper = ns.applications.PacketSinkHelper("ns3::UdpSocketFactory",
    ns.network.InetSocketAddress(ns.network.Ipv4Address.GetAny(), port))
sink_app = packet_sink_helper.Install(nodes.Get(0))
sink_app.Start(ns.core.Seconds(0.0))
sink_app.Stop(ns.core.Seconds(100.0))

onoff = ns.applications.OnOffHelper("ns3::UdpSocketFactory",
    ns.network.Address(ns.network.InetSocketAddress(sink_address, port)))
onoff.SetAttribute("DataRate", ns.core.StringValue("1Mbps"))
onoff.SetAttribute("PacketSize", ns.core.UintegerValue(64))

for i in range(1, num_nodes):
    app = onoff.Install(nodes.Get(i))
    app.Start(ns.core.Seconds(1.0 + i))
    app.Stop(ns.core.Seconds(100.0))

# Energy Model
energy = ns.energy.BasicEnergySourceHelper()
energy.Set("BasicEnergySourceInitialEnergyJ", ns.core.DoubleValue(100.0))
sources = energy.Install(nodes)

device_energy = ns.energy.WifiRadioEnergyModelHelper()
device_energy.Install(devices, sources)

# Run simulation
simulate_trlb_clustering()
ns.core.Simulator.Stop(ns.core.Seconds(100.0))
ns.core.Simulator.Run()
ns.core.Simulator.Destroy()
