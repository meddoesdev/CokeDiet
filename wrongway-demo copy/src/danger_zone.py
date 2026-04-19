"""
danger_zone.py  (v3.1 — updated)
----------------------------------
Person A — Two predictive safety features:

  1. DANGER ZONE PREDICTOR
     Dead reckoning from last confirmed wrong-way position.
     Projects next N GPS positions forward using bearing + speed.
     Person B draws this as an orange dashed line on the map.

  2. ONCOMING COLLISION RISK DETECTOR
     Checks two types of collision scenarios:
       A) wrong_way vehicle vs normal vehicle (original)
       B) wrong_way vehicle vs wrong_way vehicle (NEW in v3.1)
          — two wrong-way vehicles converging on each other,
            always classified CRITICAL regardless of distance.

Changes from v3:
  + _check_pair() helper extracted to avoid duplicated logic
  + _detect_ww_vs_normal() — original logic, now isolated
  + _detect_ww_vs_ww()     — NEW: checks all ww vehicle pairs
  + detect_collision_risks() calls both and merges results
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2, fabs

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

PROJECTION_STEPS      = 5       # future positions to predict per vehicle
PROJECTION_INTERVAL_S = 2       # seconds between projected points (match GPS rate)
COLLISION_THRESHOLD_M = 80      # metres — proximity threshold for collision flag
TIME_WINDOW_S         = 4       # seconds — max timestamp gap to count as "same moment"

LABEL_WRONG_WAY = "wrong_way"
LABEL_NORMAL    = "normal"


# ─────────────────────────────────────────────
# Haversine distance
# ─────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in metres between two GPS coordinates.
    Accurate to <0.5% at urban distances (<5km).
    """
    R = 6_371_000
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ─────────────────────────────────────────────
# Feature 1: Danger Zone Predictor
# ─────────────────────────────────────────────

def _project_position(lat: float, lon: float, bearing_deg: float,
                       speed_kmh: float, interval_s: float) -> tuple:
    """
    One dead-reckoning step. Flat-earth approximation, <0.1% error under 500m.

    Returns:
        (new_lat, new_lon) rounded to 6 decimal places
    """
    speed_ms    = speed_kmh / 3.6
    distance_m  = speed_ms * interval_s
    bearing_rad = radians(bearing_deg)

    delta_lat = (distance_m * cos(bearing_rad)) / 111_000
    delta_lon = (distance_m * sin(bearing_rad)) / (111_000 * cos(radians(lat)))

    return (round(lat + delta_lat, 6), round(lon + delta_lon, 6))


def predict_danger_zones(labeled_df: pd.DataFrame,
                          steps: int = PROJECTION_STEPS,
                          interval_s: float = PROJECTION_INTERVAL_S) -> list:
    """
    For each confirmed wrong-way vehicle, project `steps` future positions
    from its last known wrong-way point using dead reckoning.

    Returns list of prediction records — one per wrong-way vehicle.
    Each record contains a `predicted_path` array for Person B to render
    as an orange dashed polyline.
    """
    ww = labeled_df[labeled_df["label"] == LABEL_WRONG_WAY].copy()
    if ww.empty:
        return []

    ww["timestamp"] = pd.to_datetime(ww["timestamp"])
    ww.sort_values(["vehicle_id", "timestamp"], inplace=True)

    predictions = []

    for vehicle_id, group in ww.groupby("vehicle_id"):
        last    = group.iloc[-1]
        lat     = last["lat"]
        lon     = last["lon"]
        bearing = last["bearing"]
        speed   = last.get("speed_kmh", 30.0)

        if pd.isna(speed) or speed <= 0:
            speed = 30.0

        path = []
        cur_lat, cur_lon = lat, lon

        for step in range(1, steps + 1):
            cur_lat, cur_lon = _project_position(
                cur_lat, cur_lon, bearing, float(speed), interval_s
            )
            path.append({
                "lat":      cur_lat,
                "lon":      cur_lon,
                "t_plus_s": step * interval_s,
            })

        predictions.append({
            "vehicle_id":     vehicle_id,
            "from_timestamp": str(last["timestamp"]),
            "from_lat":       lat,
            "from_lon":       lon,
            "bearing":        round(float(bearing), 1),
            "speed_kmh":      round(float(speed), 1),
            "predicted_path": path,
        })

    print(f"[danger_zone] Predicted danger zones for "
          f"{len(predictions)} wrong-way vehicle(s).")
    return predictions


# ─────────────────────────────────────────────
# Feature 2: Collision Risk Detector
# ─────────────────────────────────────────────

def _are_heading_toward_each_other(bearing_a: float, bearing_b: float,
                                    tolerance: float = 60.0) -> bool:
    """
    Returns True if bearing_a and bearing_b are roughly opposing,
    meaning the two vehicles are heading toward each other.

    Head-on = bearings differ by ~180 degrees. We allow +/-60 degree tolerance.
    Deliberately generous — better to over-warn than miss a collision.
    """
    diff = fabs(bearing_a - bearing_b) % 360
    diff = min(diff, 360 - diff)
    return fabs(diff - 180) <= tolerance


def _bearing_between(lat1: float, lon1: float,
                      lat2: float, lon2: float) -> float:
    """
    Compass bearing from point 1 to point 2, in degrees (0-360).
    Used in ww-vs-ww convergence check to confirm vehicles are
    moving toward each other, not away.
    """
    lat1_r = radians(lat1)
    lat2_r = radians(lat2)
    d_lon  = radians(lon2 - lon1)

    x = sin(d_lon) * cos(lat2_r)
    y = (cos(lat1_r) * sin(lat2_r)
         - sin(lat1_r) * cos(lat2_r) * cos(d_lon))

    bearing = (360 + atan2(x, y) * 180 / 3.141592653589793) % 360
    return bearing


def _build_collision_record(row_a: pd.Series, row_b: pd.Series,
                              label_a: str, label_b: str,
                              dist_m: float, time_diff: float,
                              severity: str) -> dict:
    """
    Builds a standardised collision risk record from two vehicle rows.
    Works for both ww-vs-normal and ww-vs-ww pairs.

    Includes legacy keys (wrong_way_vehicle, normal_vehicle) so Person B's
    existing popup code doesn't need to change.
    """
    collision_type = (
        "ww_vs_normal"
        if label_a != label_b
        else "ww_vs_ww"
    )
    return {
        "vehicle_a":         row_a["vehicle_id"],
        "vehicle_b":         row_b["vehicle_id"],
        "label_a":           label_a,
        "label_b":           label_b,
        "collision_type":    collision_type,
        "timestamp":         str(row_a["timestamp"]),
        "edge_id":           row_a["edge_id"],
        "distance_m":        round(dist_m, 1),
        "time_diff_s":       round(time_diff, 1),
        "lat_a":             row_a["lat"],
        "lon_a":             row_a["lon"],
        "bearing_a":         round(float(row_a["bearing"]), 1),
        "lat_b":             row_b["lat"],
        "lon_b":             row_b["lon"],
        "bearing_b":         round(float(row_b["bearing"]), 1),
        "centroid_lat":      round((row_a["lat"] + row_b["lat"]) / 2, 6),
        "centroid_lon":      round((row_a["lon"] + row_b["lon"]) / 2, 6),
        "severity":          severity,
        # Legacy keys — keep these so Person B's popup doesn't break
        "wrong_way_vehicle": row_a["vehicle_id"],
        "normal_vehicle":    row_b["vehicle_id"],
        "ww_lat":            row_a["lat"],
        "ww_lon":            row_a["lon"],
        "ww_bearing":        round(float(row_a["bearing"]), 1),
        "normal_lat":        row_b["lat"],
        "normal_lon":        row_b["lon"],
        "normal_bearing":    round(float(row_b["bearing"]), 1),
    }


def _detect_ww_vs_normal(ww_df: pd.DataFrame,
                          nrm_df: pd.DataFrame,
                          threshold_m: float,
                          seen_pairs: set) -> list:
    """
    Original logic: wrong-way vehicle vs normal vehicle.

    All four conditions required:
      1. Same edge_id
      2. Timestamps within TIME_WINDOW_S seconds
      3. Physical distance within threshold_m metres
      4. Opposing bearings (heading toward each other)
    """
    risks = []

    for _, ww_row in ww_df.iterrows():
        same_edge = nrm_df[nrm_df["edge_id"] == ww_row["edge_id"]]
        if same_edge.empty:
            continue

        for _, nrm_row in same_edge.iterrows():
            time_diff = abs(
                (ww_row["timestamp"] - nrm_row["timestamp"]).total_seconds()
            )
            if time_diff > TIME_WINDOW_S:
                continue

            dist_m = haversine_m(
                ww_row["lat"], ww_row["lon"],
                nrm_row["lat"], nrm_row["lon"]
            )
            if dist_m > threshold_m:
                continue

            if not _are_heading_toward_each_other(
                ww_row["bearing"], nrm_row["bearing"]
            ):
                continue

            time_bucket = ww_row["timestamp"].floor("4s")
            pair_key = (
                ww_row["vehicle_id"],
                nrm_row["vehicle_id"],
                str(time_bucket)
            )
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            severity = "CRITICAL" if dist_m < threshold_m / 2 else "HIGH"

            risks.append(_build_collision_record(
                ww_row, nrm_row,
                LABEL_WRONG_WAY, LABEL_NORMAL,
                dist_m, time_diff, severity
            ))

    return risks


def _detect_ww_vs_ww(ww_df: pd.DataFrame,
                      threshold_m: float,
                      seen_pairs: set) -> list:
    """
    NEW in v3.1: wrong-way vehicle vs wrong-way vehicle.

    Two wrong-way vehicles converging is MORE dangerous than ww-vs-normal
    because neither vehicle is in a safe lane.

    Key differences from ww-vs-normal:
      - Does NOT require same edge_id — two wrong-way vehicles on adjacent
        or intersecting edges can still collide. Proximity alone is enough.
      - Does NOT require opposing bearings — any converging trajectory is
        dangerous. We instead check that vehicle A is moving TOWARD vehicle B
        (not away) using a bearing-to-target comparison.
      - Severity is always CRITICAL — no safe escape path exists for either.
    """
    risks = []
    ww_vehicles = list(ww_df["vehicle_id"].unique())

    for i, veh_a_id in enumerate(ww_vehicles):
        for veh_b_id in ww_vehicles[i + 1:]:
            group_a = ww_df[ww_df["vehicle_id"] == veh_a_id]
            group_b = ww_df[ww_df["vehicle_id"] == veh_b_id]

            for _, row_a in group_a.iterrows():
                time_diffs = (
                    group_b["timestamp"] - row_a["timestamp"]
                ).dt.total_seconds().abs()
                close_in_time = group_b[time_diffs <= TIME_WINDOW_S]

                for _, row_b in close_in_time.iterrows():
                    dist_m = haversine_m(
                        row_a["lat"], row_a["lon"],
                        row_b["lat"], row_b["lon"]
                    )
                    if dist_m > threshold_m:
                        continue

                    # Check vehicle A is moving toward vehicle B, not away.
                    # Compute the bearing FROM A's position TO B's position,
                    # then compare to A's actual heading.
                    # If the angle between them is > 90 degrees, A is moving away.
                    bearing_a_to_b = _bearing_between(
                        row_a["lat"], row_a["lon"],
                        row_b["lat"], row_b["lon"]
                    )
                    angle_diff = fabs(row_a["bearing"] - bearing_a_to_b) % 360
                    angle_diff = min(angle_diff, 360 - angle_diff)
                    if angle_diff > 90:
                        continue

                    time_diff = fabs(
                        (row_a["timestamp"] - row_b["timestamp"]).total_seconds()
                    )

                    time_bucket = row_a["timestamp"].floor("4s")
                    pair_key = (
                        min(veh_a_id, veh_b_id),
                        max(veh_a_id, veh_b_id),
                        str(time_bucket)
                    )
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    risks.append(_build_collision_record(
                        row_a, row_b,
                        LABEL_WRONG_WAY, LABEL_WRONG_WAY,
                        dist_m, time_diff, "CRITICAL"
                    ))

    return risks


def detect_collision_risks(labeled_df: pd.DataFrame,
                             threshold_m: float = COLLISION_THRESHOLD_M) -> list:
    """
    Master collision detector. Runs both sub-detectors and merges results.

    Scenario A — ww_vs_normal:
      wrong-way vehicle heading toward a normal vehicle on the same edge.
      Severity: CRITICAL if <40m, HIGH if 40-80m.

    Scenario B — ww_vs_ww:
      Two wrong-way vehicles converging on each other within threshold_m.
      Severity: always CRITICAL.

    Returns:
        List of collision risk dicts, sorted closest-first (most urgent first).
        Each dict contains both new keys (vehicle_a, vehicle_b, collision_type)
        and legacy keys (wrong_way_vehicle, normal_vehicle) for backwards
        compatibility with Person B's existing popup rendering code.
    """
    labeled_df = labeled_df.copy()
    labeled_df["timestamp"] = pd.to_datetime(labeled_df["timestamp"])

    ww_df  = labeled_df[labeled_df["label"] == LABEL_WRONG_WAY].copy()
    nrm_df = labeled_df[labeled_df["label"] == LABEL_NORMAL].copy()

    if ww_df.empty:
        print("[danger_zone] No wrong-way vehicles — skipping collision detection.")
        return []

    seen_pairs      = set()
    collision_risks = []

    # Sub-detector A: ww vs normal
    if not nrm_df.empty:
        ww_vs_normal = _detect_ww_vs_normal(
            ww_df, nrm_df, threshold_m, seen_pairs
        )
        collision_risks.extend(ww_vs_normal)
        print(f"[danger_zone] ww-vs-normal: {len(ww_vs_normal)} event(s)")
    else:
        print("[danger_zone] ww-vs-normal: skipped (no normal vehicles)")

    # Sub-detector B: ww vs ww (NEW)
    if ww_df["vehicle_id"].nunique() >= 2:
        ww_vs_ww = _detect_ww_vs_ww(ww_df, threshold_m, seen_pairs)
        collision_risks.extend(ww_vs_ww)
        print(f"[danger_zone] ww-vs-ww:     {len(ww_vs_ww)} event(s)")
    else:
        print("[danger_zone] ww-vs-ww:     skipped "
              "(need 2+ wrong-way vehicles)")

    collision_risks.sort(key=lambda x: x["distance_m"])

    n      = len(collision_risks)
    n_crit = sum(1 for r in collision_risks if r["severity"] == "CRITICAL")
    print(f"[danger_zone] Total: {n} collision risk(s) — "
          f"CRITICAL: {n_crit}, HIGH: {n - n_crit}")

    return collision_risks