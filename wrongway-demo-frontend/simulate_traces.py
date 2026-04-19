import osmnx as ox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
import os

# ── GRAPH ────────────────────────────────────────────────────
G = ox.graph_from_point((28.6315, 77.2167), dist=800, network_type="drive")
print("Nodes:", len(G.nodes), "| Edges:", len(G.edges))

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/output", exist_ok=True)

# ── HELPERS ──────────────────────────────────────────────────
def compute_bearing(lat1, lon1, lat2, lon2):
    d_lon = math.radians(lon2 - lon1)
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(d_lon)
    return round((math.degrees(math.atan2(x, y)) + 360) % 360, 1)

def interpolate_segment(lat1, lon1, lat2, lon2, n_points=6):
    """Generate n_points evenly spaced between two coordinates."""
    lats = np.linspace(lat1, lat2, n_points)
    lons = np.linspace(lon1, lon2, n_points)
    return list(zip(lats, lons))

def densify_coords(coords, points_per_segment=6):
    """
    Take a sparse list of (lat,lon) nodes and interpolate
    points_per_segment GPS points between each pair.
    This ensures multiple consecutive points on each segment.
    """
    dense = []
    for i in range(len(coords) - 1):
        segment_pts = interpolate_segment(
            coords[i][0], coords[i][1],
            coords[i+1][0], coords[i+1][1],
            n_points=points_per_segment
        )
        # Avoid duplicating the junction point
        dense += segment_pts[:-1]
    dense.append(coords[-1])
    return dense

def get_route_coords(G, orig_node, dest_node):
    route = ox.shortest_path(G, orig_node, dest_node)
    if route is None or len(route) < 4:
        return None
    return [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]

def build_trace(coords, vehicle_id, start_time, speed_profile='normal'):
    rows = []
    n = len(coords)
    for i, (lat, lon) in enumerate(coords):
        timestamp = start_time + timedelta(seconds=i*2)
        if i < n - 1:
            bearing = compute_bearing(lat, lon, coords[i+1][0], coords[i+1][1])
        else:
            bearing = compute_bearing(coords[i-1][0], coords[i-1][1], lat, lon)

        if speed_profile == 'normal':
            speed = round(np.random.uniform(30, 50), 1)
        elif speed_profile == 'wrongway':
            speed = round(np.random.uniform(25, 40), 1)
        elif speed_profile == 'uturn':
            if n // 3 < i < 2 * n // 3:
                speed = round(np.random.uniform(5, 8), 1)
            elif i < n // 3:
                speed = round(np.random.uniform(10, 15), 1)
            else:
                speed = round(np.random.uniform(30, 40), 1)
        elif speed_profile == 'multivehicle':
            speed = round(np.random.uniform(25, 40), 1)
        else:
            speed = round(np.random.uniform(30, 50), 1)

        rows.append({
            'vehicle_id': vehicle_id,
            'lat': round(lat, 7),
            'lon': round(lon, 7),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'bearing': bearing,
            'speed_kmh': speed
        })
    return rows

# ── FIND VALID ROUTE ─────────────────────────────────────────
nodes = list(G.nodes)
sparse_coords = None

for i in range(len(nodes)):
    for j in range(i + 10, min(i + 80, len(nodes))):
        coords = get_route_coords(G, nodes[i], nodes[j])
        if coords and len(coords) >= 6:
            sparse_coords = coords
            print(f"Found route: nodes[{i}] -> nodes[{j}], "
                  f"{len(coords)} sparse points")
            break
    if sparse_coords:
        break

if not sparse_coords:
    print("ERROR: No valid route found. Try increasing dist=")
    exit()

# ── DENSIFY — 6 GPS points per road segment ──────────────────
# This is the key fix: each road segment now has 6 consecutive
# points so Person A's run-length detector sees sustained runs
normal_coords  = densify_coords(sparse_coords, points_per_segment=6)
wrongway_coords = normal_coords[::-1]
uturn_coords   = normal_coords + normal_coords[::-1]

print(f"Dense normal route:    {len(normal_coords)} points")
print(f"Dense wrong-way route: {len(wrongway_coords)} points")
print(f"Dense u-turn route:    {len(uturn_coords)} points")

# ── BUILD ALL SCENARIOS ───────────────────────────────────────
start = datetime(2024, 1, 1, 9, 0, 0)
rows = []

# Scenario A — basic wrong-way
rows += build_trace(normal_coords,   'veh_001',      start, 'normal')
rows += build_trace(wrongway_coords, 'veh_wrongway', start, 'wrongway')

# Scenario B — U-turn false positive (speed filter should suppress)
rows += build_trace(uturn_coords, 'veh_uturn',
                    start + timedelta(seconds=10), 'uturn')

# Scenario C — multi-vehicle pile-on on same segment
rows += build_trace(wrongway_coords, 'veh_003',
                    start + timedelta(seconds=4), 'multivehicle')

df = pd.DataFrame(
    rows,
    columns=['vehicle_id','lat','lon','timestamp','bearing','speed_kmh']
)
df.to_csv("data/raw/gps_traces.csv", index=False)

print(f"\nTotal rows saved: {len(df)}")
print(df.groupby('vehicle_id').size().to_string())
print("\nSample (veh_wrongway first 8 rows):")
print(df[df.vehicle_id=='veh_wrongway'].head(8).to_string(index=False))