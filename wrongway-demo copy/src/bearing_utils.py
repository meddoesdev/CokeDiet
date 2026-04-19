"""
bearing_utils.py
----------------
Pure utility functions for bearing/angle math.
No OSM or pandas dependencies — easy to unit test in isolation.
"""

import numpy as np


def angular_difference(bearing_a: float, bearing_b: float) -> float:
    """
    Returns the smallest angular difference between two bearings (0–360°).
    Handles wraparound correctly: e.g. diff(5, 355) = 10, not 350.

    Args:
        bearing_a: First bearing in degrees (0–360)
        bearing_b: Second bearing in degrees (0–360)

    Returns:
        Angular difference in degrees, always in range [0, 180]
    """
    diff = abs(bearing_a - bearing_b) % 360
    return min(diff, 360 - diff)


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the compass bearing (degrees, 0–360) from point 1 to point 2.
    Used as a fallback if Person B's CSV doesn't include bearing.

    Args:
        lat1, lon1: Start point (decimal degrees)
        lat2, lon2: End point (decimal degrees)

    Returns:
        Bearing in degrees, clockwise from north (0–360)
    """
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    d_lon  = np.radians(lon2 - lon1)

    x = np.sin(d_lon) * np.cos(lat2_r)
    y = np.cos(lat1_r) * np.sin(lat2_r) - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(d_lon)

    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def reverse_bearing(bearing: float) -> float:
    """Returns the opposite direction of a bearing."""
    return (bearing + 180) % 360


def is_wrong_way(vehicle_bearing: float, road_bearing: float,
                 threshold: float = 120.0) -> bool:
    """
    Returns True if the vehicle is travelling against the road's allowed direction.

    Logic: if the angular diff between vehicle heading and road bearing
    exceeds `threshold`, the vehicle is likely going the wrong way.

    A threshold of 120° gives a ±60° tolerance around the opposing direction
    (i.e., we only flag if clearly opposing, not just at an angle).

    Args:
        vehicle_bearing: Heading of the vehicle (degrees, 0–360)
        road_bearing:    Allowed direction of the road segment (degrees, 0–360)
        threshold:       Angular difference to trigger a flag (default 120°)

    Returns:
        True if wrong-way, False otherwise
    """
    diff = angular_difference(vehicle_bearing, road_bearing)
    return diff > threshold