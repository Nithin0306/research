#!/usr/bin/env python3
"""
Pure‑Python HEED simulation rewritten from the original NS‑3–dependent version.
All NS‑3 bindings have been removed, so the script can be executed with a plain
Python 3 interpreter.  Functionality and data‑loading semantics remain the same.
"""

import sys
import os
import random
from collections import defaultdict
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
#  Global simulation constants (same values as in the original code)
# --------------------------------------------------------------------------------------

SIMULATION_SEED: int = 42
AREA_SIZE: int = 100  # m
INIT_ENERGY: float = 2.0  # J
BASE_STATION_POS = (AREA_SIZE / 2, 110.0)

TX_ENERGY_PER_BIT: float = 50e-9  # J / bit
RX_ENERGY_PER_BIT: float = 50e-9  # J / bit
FS_ENERGY: float = 10e-12
MP_ENERGY: float = 0.0013e-12
PACKET_SIZE: int = 4000  # bit
IDLE_ENERGY: float = 1e-4  # J / s
TIME_PER_BIT: float = 1e-6  # s
DEADLINE: float = 0.1  # s

random.seed(SIMULATION_SEED)
np.random.seed(SIMULATION_SEED)

# --------------------------------------------------------------------------------------
#  Node Model
# --------------------------------------------------------------------------------------

class WSNNode:
    """Wireless‑sensor‑network node (pure Python)."""

    __slots__ = (
        "id",
        "x",
        "y",
        "energy",
        "battery_level",
        "sensor_type",
        "is_CH",
        "cluster_head",
        "prev_ch_id",
        "alive",
        "distance_to_bs",
        # runtime stats
        "temperature",
        "pressure",
        "sensor_cycle",
        "packets_sent",
        "packets_received",
    )

    def __init__(self, node_id: int, x: float, y: float, initial_battery: float, sensor_type: str):
        self.id = node_id
        self.x = x
        self.y = y
        self.energy = INIT_ENERGY
        self.battery_level = float(initial_battery)
        self.sensor_type = sensor_type
        self.is_CH = False
        self.cluster_head = None
        self.prev_ch_id = None
        self.alive = True
        self.distance_to_bs = ((self.x - BASE_STATION_POS[0]) ** 2 + (self.y - BASE_STATION_POS[1]) ** 2) ** 0.5

        # telemetry
        self.temperature = 0.0
        self.pressure = 0.0
        self.sensor_cycle = 0
        self.packets_sent = 0
        self.packets_received = 0

    # ------------------------------------------------------------------ utility

    def is_alive(self) -> bool:
        return self.alive and self.energy > 0.0 and self.battery_level > 0.0

    def distance_to(self, other: "WSNNode") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    # ------------------------------------------------------------------ state updates

    def update_battery(self, new_level: float) -> None:
        self.battery_level = max(0.0, float(new_level))
        self.energy = INIT_ENERGY * (self.battery_level / 100.0)
        if self.battery_level <= 0.0:
            self.alive = False

    def update_sensor_data(self, temp_c: float, hpa_div_4: float, sensor_cycle: int) -> None:
        self.temperature = float(temp_c)
        self.pressure = float(hpa_div_4)
        self.sensor_cycle = int(sensor_cycle)

    # ------------------------------------------------------------------ energy model

    def consume_transmission_energy(self, distance: float) -> float:
        """Consume energy for transmitting PACKET_SIZE bits over *distance* metres.
        Returns the transmission latency in seconds (for the one hop).
        """
        if not self.is_alive():
            return 0.0

        d_threshold = 75.0
        if distance < d_threshold:
            energy = TX_ENERGY_PER_BIT * PACKET_SIZE + FS_ENERGY * PACKET_SIZE * (distance ** 2)
        else:
            energy = TX_ENERGY_PER_BIT * PACKET_SIZE + MP_ENERGY * PACKET_SIZE * (distance ** 4)

        self.energy -= energy
        self.battery_level -= (energy / INIT_ENERGY) * 5.0

        if self.energy <= 0.0 or self.battery_level <= 0.0:
            self.alive = False
        return PACKET_SIZE * TIME_PER_BIT

    def consume_reception_energy(self) -> float:
        if not self.is_alive():
            return 0.0
        energy = RX_ENERGY_PER_BIT * PACKET_SIZE
        self.energy -= energy
        self.battery_level -= (energy / INIT_ENERGY) * 2.0
        if self.energy <= 0.0 or self.battery_level <= 0.0:
            self.alive = False
        return PACKET_SIZE * TIME_PER_BIT

    def consume_idle_energy(self) -> None:
        if not self.is_alive():
            return
        self.energy -= IDLE_ENERGY
        self.battery_level -= 0.01
        if self.energy <= 0.0 or self.battery_level <= 0.0:
            self.alive = False

# --------------------------------------------------------------------------------------
#  HEED Simulation (pure Python)
# --------------------------------------------------------------------------------------

class HEEDSimulation:
    """Pure‑Python HEED clustering simulation compatible with the original interface."""

    # ------------- construction / initialisation ------------------------------------------------
    def __init__(self):
        self.nodes: dict[int, WSNNode] = {}
        self.round_num: int = 0
        self.total_transmissions: int = 0

        self.stats: dict[str, list] = {
            "alive_nodes": [],
            "total_energy": [],
            "chs_per_round": [],
            "delivered_packets": [],
            "dropped_packets": [],
            "cluster_switches": [],
            "load_distribution": [],
            "round_delay": [],
            "met_deadline_packets": [],
            "missed_deadline_packets": [],
            "real_time_ratio": [],
            "pdr_over_time": [],
            "first_death": None,
            "last_death": None,
        }

    # ------------- dataset ------------------------------------------------

    @staticmethod
    def load_and_validate_dataset(file_path: str) -> pd.DataFrame | None:
        if not os.path.exists(file_path):
            print(f"Dataset file not found: {file_path}")
            return None
        df = pd.read_csv(file_path)
        required_cols = [
            "sort_id",
            "date_d_m_y",
            "time",
            "sensor_id",
            "sensor_type",
            "temp_C",
            "hpa_div_4",
            "batterylevel",
            "sensor_cycle",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print("Missing required columns:", missing)
            return None
        # datetime merge & clean
        df["datetime"] = pd.to_datetime(
            df["date_d_m_y"] + " " + df["time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
        )
        df = df.dropna(subset=["datetime"])
        numeric_cols = ["temp_C", "hpa_div_4", "batterylevel", "sensor_cycle"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=numeric_cols)
        df = df[(df["batterylevel"] >= 0) & (df["batterylevel"] <= 100)]
        df = df.sort_values(["datetime", "sensor_id"]).reset_index(drop=True)
        return df

    # ------------- node initialisation ------------------------------------------------

    def initialise_nodes(self, df: pd.DataFrame) -> None:
        sensor_info = (
            df.groupby("sensor_id")
            .agg(
                sensor_type=("sensor_type", "first"),
                batterylevel=("batterylevel", "first"),
                temp_C=("temp_C", "first"),
                hpa_div_4=("hpa_div_4", "first"),
                sensor_cycle=("sensor_cycle", "first"),
            )
            .reset_index()
        )

        grid_size = int(np.ceil(np.sqrt(len(sensor_info))))
        cell_size = AREA_SIZE / grid_size

        for i, row in sensor_info.iterrows():
            gx, gy = i % grid_size, i // grid_size
            x = gx * cell_size + random.uniform(5.0, cell_size - 5.0)
            y = gy * cell_size + random.uniform(5.0, cell_size - 5.0)
            node = WSNNode(
                int(row.sensor_id),
                float(max(5.0, min(x, AREA_SIZE - 5.0))),
                float(max(5.0, min(y, AREA_SIZE - 5.0))),
                float(row.batterylevel),
                str(row.sensor_type),
            )
            node.update_sensor_data(row.temp_C, row.hpa_div_4, row.sensor_cycle)
            self.nodes[node.id] = node

    # ------------- helpers ------------------------------------------------

    def create_time_windows(self, df: pd.DataFrame, minutes: int = 30):
        df["time_window"] = df["datetime"].dt.floor(f"{minutes}min")
        return list(df.groupby("time_window"))

    @staticmethod
    def heed_cluster_formation(alive_nodes: list[WSNNode]):
        cluster_heads: list[WSNNode] = []
        for n in alive_nodes:
            n.is_CH = False
        for n in alive_nodes:
            energy_factor = n.energy / INIT_ENERGY
            battery_factor = n.battery_level / 100.0
            cycle_factor = max(0.1, 1.0 - (n.sensor_cycle / 1000.0))
            p = min(1.0, 0.08 * energy_factor * battery_factor * cycle_factor)
            if random.random() < p:
                n.is_CH = True
                cluster_heads.append(n)
        if not cluster_heads:
            alive_nodes.sort(
                key=lambda _n: _n.energy * _n.battery_level * (1000 - _n.sensor_cycle),
                reverse=True,
            )
            cluster_heads = alive_nodes[: max(1, len(alive_nodes) // 10)]
            for ch in cluster_heads:
                ch.is_CH = True
        return cluster_heads

    # ------------- transmission ------------------------------------------------

    def transmit(self, sender: WSNNode, receiver: WSNNode):
        if not (sender.is_alive() and receiver.is_alive()):
            return False, 0.0
        d = sender.distance_to(receiver)
        tx_delay = sender.consume_transmission_energy(d)
        rx_delay = receiver.consume_reception_energy()
        sender.packets_sent += 1
        if sender.is_alive() and receiver.is_alive():
            receiver.packets_received += 1
            return True, tx_delay + rx_delay
        return False, 0.0

    # ------------- round simulation ------------------------------------------------

    def simulate_round(self, window_df: pd.DataFrame, cluster_heads: list[WSNNode]):
        delivered = dropped = switches = met_rt = miss_rt = 0
        total_delay = 0.0
        load = defaultdict(int)

        for _idx, rec in window_df.iterrows():
            sid = rec.sensor_id
            self.total_transmissions += 1
            node = self.nodes.get(sid)
            if node is None or not node.is_alive():
                dropped += 1
                continue

            node.update_sensor_data(rec.temp_C, rec.hpa_div_4, rec.sensor_cycle)

            # CH case ────────────────────────────────────────────────────
            if node.is_CH:
                delay = node.consume_transmission_energy(node.distance_to_bs)
                if node.is_alive():
                    delivered += 1
                    total_delay += delay
                    load[node.id] += 1
                    (met_rt if delay <= DEADLINE else miss_rt)  # noqa: B018 (intentional noop)
                    if delay <= DEADLINE:
                        met_rt += 1
                    else:
                        miss_rt += 1
                else:
                    dropped += 1
                continue

            # Non‑CH case ────────────────────────────────────────────────
            alive_chs = [c for c in cluster_heads if c.is_alive()]
            if not alive_chs:
                dropped += 1
                continue
            closest_ch = min(alive_chs, key=lambda ch: node.distance_to(ch))
            if node.prev_ch_id is not None and node.prev_ch_id != closest_ch.id:
                switches += 1
            node.prev_ch_id = closest_ch.id
            success, d1 = self.transmit(node, closest_ch)
            if not success:
                dropped += 1
                continue
            load[closest_ch.id] += 1
            d2 = closest_ch.consume_transmission_energy(closest_ch.distance_to_bs)
            if closest_ch.is_alive():
                delay_total = d1 + d2
                delivered += 1
                total_delay += delay_total
                if delay_total <= DEADLINE:
                    met_rt += 1
                else:
                    miss_rt += 1
            else:
                dropped += 1

        load_std = np.std(list(load.values()) or [0])
        return {
            "delivered": delivered,
            "dropped": dropped,
            "switches": switches,
            "met_deadline": met_rt,
            "missed_deadline": miss_rt,
            "total_delay": total_delay,
            "load_std": load_std,
        }

    # ------------- statistics ------------------------------------------------

    def update_statistics(self, res: dict, cluster_heads: list[WSNNode]):
        alive = sum(n.is_alive() for n in self.nodes.values())
        total_e = sum(n.energy for n in self.nodes.values() if n.is_alive())
        self.stats["alive_nodes"].append(alive)
        self.stats["total_energy"].append(total_e)
        self.stats["chs_per_round"].append(len(cluster_heads))
        self.stats["delivered_packets"].append(res["delivered"])
        self.stats["dropped_packets"].append(res["dropped"])
        self.stats["cluster_switches"].append(res["switches"])
        self.stats["round_delay"].append(res["total_delay"])
        self.stats["met_deadline_packets"].append(res["met_deadline"])
        self.stats["missed_deadline_packets"].append(res["missed_deadline"])
        self.stats["load_distribution"].append(res["load_std"])

        tot = res["delivered"] + res["dropped"]
        self.stats["pdr_over_time"].append(res["delivered"] / tot if tot else 0.0)
        self.stats["real_time_ratio"].append(
            res["met_deadline"] / res["delivered"] if res["delivered"] else 0.0
        )

        if alive < len(self.nodes) and self.stats["first_death"] is None:
            self.stats["first_death"] = self.round_num
        if alive == 0 and self.stats["last_death"] is None:
            self.stats["last_death"] = self.round_num

    # ------------- driver ------------------------------------------------

    def run(self, dataset_path: str):
        df = self.load_and_validate_dataset(dataset_path)
        if df is None:
            return None
        self.initialise_nodes(df)
        windows = self.create_time_windows(df)
        for _, window_df in windows:
            self.round_num += 1
            for sid, grp in window_df.groupby("sensor_id"):
                n = self.nodes.get(sid)
                if n:
                    n.update_battery(grp.batterylevel.iloc[-1])
            alive_nodes = [n for n in self.nodes.values() if n.is_alive()]
            if not alive_nodes:
                self.stats["last_death"] = self.round_num
                break
            for n in alive_nodes:
                n.consume_idle_energy()
            chs = self.heed_cluster_formation(alive_nodes)
            res = self.simulate_round(window_df, chs)
            self.update_statistics(res, chs)
        return {
            "stats": self.stats,
            "total_rounds": self.round_num,
            "total_nodes": len(self.nodes),
            "final_alive_nodes": sum(n.is_alive() for n in self.nodes.values()),
        }

    # ------------- reporting ------------------------------------------------

    def print_summary(self, results: dict):
        s = results["stats"]
        print("\n================ HEED SIMULATION RESULTS ================")
        print(f"Total nodes          : {results['total_nodes']}")
        print(f"Simulation rounds    : {results['total_rounds']}")
        print(f"Final alive nodes    : {results['final_alive_nodes']}")
        print("---------------------------------------------------------")
        delivered = sum(s["delivered_packets"])
        dropped = sum(s["dropped_packets"])
        total = delivered + dropped
        if total:
            print(f"Overall PDR          : {delivered / total:.4f}")
        if s["first_death"] is not None:
            print(f"First node death @   : round {s['first_death']}")
        if s["last_death"] is not None:
            print(f"Last node death @    : round {s['last_death']}")
        print("=========================================================")

# --------------------------------------------------------------------------------------
#  main
# --------------------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = input("Dataset path: ").strip().strip("\'\"")
    if not os.path.exists(dataset_path):
        print("File not found:", dataset_path)
        sys.exit(1)
    sim = HEEDSimulation()
    t0 = time.time()
    res = sim.run(dataset_path)
    if res is None:
        print("Simulation failed – invalid dataset.")
        sys.exit(1)
    sim.print_summary(res)
    print(f"Completed in {time.time() - t0:.2f} s")

if __name__ == "__main__":
    main()
