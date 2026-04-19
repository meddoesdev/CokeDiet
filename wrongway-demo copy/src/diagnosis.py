import pandas as pd
from road_graph import load_graph, snap_to_road
from bearing_utils import angular_difference

df = pd.read_csv("data/raw/gps_traces.csv")
G = load_graph()

print("vehicle_bearing | road_bearing | angular_diff")
for _, row in df.iterrows():
    snap = snap_to_road(G, row["lat"], row["lon"])
    if snap:
        u, v, key, road_bearing, dist = snap
        diff = angular_difference(row["bearing"], road_bearing)
        print(f"  {row['bearing']:>7.1f}°      |   {road_bearing:>7.1f}°    |   {diff:.1f}°   {row['vehicle_id']}")