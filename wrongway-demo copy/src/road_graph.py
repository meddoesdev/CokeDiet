"""
road_graph.py  (v2 — upgraded)
--------------------------------
Person A — OSM Graph + Road Snapping

Changes from v1:
  + snap_to_road now returns 5-tuple: (u, v, key, road_bearing, snap_distance_m)
    so detector.py can feed snap_distance into confidence scoring.
  + snap_distance computed properly in metres using a degree-to-metres conversion.
"""

import os
import numpy as np
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString

PLACE_NAME      = "Connaught Place, New Delhi, India"   # ← change if needed
GRAPH_SAVE_PATH = "data/raw/road_graph.graphml"
MAX_SNAP_DISTANCE_M = 30


def load_graph(place: str = PLACE_NAME,
               save_path: str = GRAPH_SAVE_PATH) -> nx.MultiDiGraph:
    if os.path.exists(save_path):
        print(f"[road_graph] Loading cached graph from {save_path}")
        G = ox.load_graphml(save_path)
    else:
        print(f"[road_graph] Downloading OSM graph for '{place}'...")
        G = ox.graph_from_place(place, network_type="drive")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ox.save_graphml(G, save_path)
        print(f"[road_graph] Saved to {save_path}")

    G = ox.bearing.add_edge_bearings(G)
    return G


def get_edge_bearing(G: nx.MultiDiGraph, u: int, v: int, key: int = 0) -> float:
    edge_data = G.edges[u, v, key]
    if "bearing" in edge_data:
        return edge_data["bearing"]
    from bearing_utils import compute_bearing
    u_data = G.nodes[u]
    v_data = G.nodes[v]
    return compute_bearing(u_data["y"], u_data["x"], v_data["y"], v_data["x"])


def _degrees_to_metres(deg: float, lat: float) -> float:
    """Rough conversion: degree distance → metres at a given latitude."""
    metres_per_deg_lat = 111_000
    metres_per_deg_lon = 111_000 * np.cos(np.radians(lat))
    # deg is in the lon direction dominantly — use conservative average
    return deg * ((metres_per_deg_lat + metres_per_deg_lon) / 2)


def snap_to_road(G: nx.MultiDiGraph, lat: float, lon: float):
    """
    Find the nearest road segment to a GPS point.

    Returns:
        (u, v, key, road_bearing, snap_distance_m)  — 5-tuple
        or None if no segment within MAX_SNAP_DISTANCE_M
    """
    try:
        u, v, key = ox.nearest_edges(G, X=lon, Y=lat)

        point     = Point(lon, lat)
        edge_data = G.edges[u, v, key]

        if "geometry" in edge_data:
            road_geom = edge_data["geometry"]
        else:
            u_node = G.nodes[u]
            v_node = G.nodes[v]
            road_geom = LineString(
                [(u_node["x"], u_node["y"]), (v_node["x"], v_node["y"])]
            )

        dist_deg = point.distance(road_geom)
        dist_m   = _degrees_to_metres(dist_deg, lat)

        if dist_m > MAX_SNAP_DISTANCE_M:
            return None

        road_bearing = get_edge_bearing(G, u, v, key)
        return (u, v, key, road_bearing, dist_m)   # ← now 5-tuple

    except Exception as e:
        print(f"[snap_to_road] Error at ({lat}, {lon}): {e}")
        return None


if __name__ == "__main__":
    G = load_graph()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    node = list(G.nodes)[0]
    result = snap_to_road(G, G.nodes[node]["y"], G.nodes[node]["x"])
    print(f"Snap test: {result}")