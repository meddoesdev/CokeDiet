"""
segment_aggregator.py  (v3 — upgraded)
----------------------------------------
Changes from v2:
  + Imports and calls danger_zone.predict_danger_zones()
  + Imports and calls danger_zone.detect_collision_risks()
  + Both outputs added to segment_alerts.json under new keys:
      "danger_zones"     — predicted paths per wrong-way vehicle
      "collision_risks"  — oncoming vehicle proximity alerts
  + Summary block now includes collision_risk_count
"""

import pandas as pd
import numpy as np
import json
import os

from danger_zone import predict_danger_zones, detect_collision_risks

ALERT_OUTPUT_PATH   = "data/output/segment_alerts.json"
POINTS_OUTPUT_PATH  = "data/output/labeled_traces.csv"
TIME_WINDOW_MINUTES = 10


def aggregate_segments(labeled_df: pd.DataFrame,
                        time_window_min: int = TIME_WINDOW_MINUTES) -> pd.DataFrame:
    ww = labeled_df[labeled_df["label"] == "wrong_way"].copy()
    if ww.empty:
        print("[aggregator] No wrong_way points to aggregate.")
        return pd.DataFrame()

    ww["timestamp"] = pd.to_datetime(ww["timestamp"])
    ww["time_bucket"] = ww["timestamp"].dt.floor(f"{time_window_min}min")
    grouped = ww.groupby(["edge_id", "time_bucket"], sort=False)

    records = []
    for (edge_id, time_bucket), group in grouped:
        records.append({
            "edge_id":           edge_id,
            "time_bucket":       str(time_bucket),
            "point_count":       len(group),
            "vehicle_count":     group["vehicle_id"].nunique(),
            "vehicle_ids":       list(group["vehicle_id"].unique()),
            "mean_confidence":   round(group["confidence"].mean(), 1),
            "max_confidence":    round(group["confidence"].max(), 1),
            "mean_angular_diff": round(group["angular_diff"].mean(), 1),
            "first_seen":        str(group["timestamp"].min()),
            "last_seen":         str(group["timestamp"].max()),
            "centroid_lat":      round(group["lat"].mean(), 6),
            "centroid_lon":      round(group["lon"].mean(), 6),
            "risk_level":        _risk_level(group),
        })

    result = pd.DataFrame(records)
    result.sort_values("mean_confidence", ascending=False, inplace=True)
    return result


def _risk_level(group: pd.DataFrame) -> str:
    vehicles  = group["vehicle_id"].nunique()
    mean_conf = group["confidence"].mean()
    if vehicles >= 2 or mean_conf >= 80:
        return "HIGH"
    elif mean_conf >= 50:
        return "MEDIUM"
    else:
        return "LOW"


def build_alert_popup_data(labeled_df: pd.DataFrame) -> list:
    ww = labeled_df[labeled_df["label"] == "wrong_way"].copy()
    ww["timestamp"] = pd.to_datetime(ww["timestamp"])
    ww.sort_values(["vehicle_id", "timestamp"], inplace=True)

    incidents = []

    for vehicle_id, group in ww.groupby("vehicle_id", sort=False):
        group = group.reset_index(drop=True)
        group["time_gap"]    = group["timestamp"].diff().dt.total_seconds().fillna(0)
        group["incident_id"] = (group["time_gap"] > 10).cumsum()

        for inc_id, inc_group in group.groupby("incident_id"):
            incidents.append({
                "vehicle_id":       vehicle_id,
                "incident_index":   int(inc_id),
                "start_time":       str(inc_group["timestamp"].iloc[0]),
                "end_time":         str(inc_group["timestamp"].iloc[-1]),
                "duration_seconds": int(
                    (inc_group["timestamp"].iloc[-1] -
                     inc_group["timestamp"].iloc[0]).total_seconds()
                ),
                "point_count":       len(inc_group),
                "mean_confidence":   round(inc_group["confidence"].mean(), 1),
                "max_confidence":    round(inc_group["confidence"].max(), 1),
                "mean_angular_diff": round(inc_group["angular_diff"].mean(), 1),
                "edge_ids":          list(inc_group["edge_id"].dropna().unique()),
                "centroid_lat":      round(inc_group["lat"].mean(), 6),
                "centroid_lon":      round(inc_group["lon"].mean(), 6),
                # Confidence score component breakdown — for popup bar chart
                "confidence_components": _confidence_components(inc_group),
                "points": [
                    {
                        "lat":          row["lat"],
                        "lon":          row["lon"],
                        "timestamp":    str(row["timestamp"]),
                        "bearing":      row["bearing"],
                        "confidence":   row["confidence"],
                        "angular_diff": row["angular_diff"],
                    }
                    for _, row in inc_group.iterrows()
                ],
            })

    return incidents


def _confidence_components(inc_group: pd.DataFrame) -> dict:
    """
    Breaks the mean confidence score into its three components
    for the popup bar chart (Person B renders this).

    Returns the mean contribution of each component across the incident.
    """
    angular_diff  = inc_group["angular_diff"].mean()
    run_length    = inc_group["run_length"].mean()
    snap_dist     = inc_group["snap_distance_m"].mean() if "snap_distance_m" in inc_group.columns else 15.0

    score_A = min(max((angular_diff - 120) / 60 * 50, 0), 50)
    score_B = min(run_length / 10, 1) * 30
    score_C = max(0, 1 - (snap_dist or 15) / 30) * 20

    return {
        "angular":     round(score_A, 1),
        "persistence": round(score_B, 1),
        "snap_quality": round(score_C, 1),
        "total":       round(min(score_A + score_B + score_C, 100), 1),
    }


def export_for_personB(labeled_df: pd.DataFrame,
                        alert_path: str = ALERT_OUTPUT_PATH,
                        points_path: str = POINTS_OUTPUT_PATH):
    """
    Master export. Writes labeled_traces.csv + segment_alerts.json.

    NEW in v3 — JSON now contains two additional top-level keys:
      "danger_zones"    — predicted paths for each wrong-way vehicle
      "collision_risks" — timestep-level oncoming vehicle proximity alerts
    """
    # Full labeled CSV
    os.makedirs(os.path.dirname(points_path), exist_ok=True)
    labeled_df.to_csv(points_path, index=False)
    print(f"[aggregator] Labeled CSV → {points_path}")

    # Segment alerts
    segment_df = aggregate_segments(labeled_df)
    incidents  = build_alert_popup_data(labeled_df)

    # Heatmap
    ww = labeled_df[labeled_df["label"] == "wrong_way"]
    heatmap_data = [
        {"lat": row["lat"], "lon": row["lon"], "weight": row["confidence"] / 100.0}
        for _, row in ww.iterrows()
    ]

    # ── NEW: Danger zones + collision risks ──
    danger_zones     = predict_danger_zones(labeled_df)
    collision_risks  = detect_collision_risks(labeled_df)

    output = {
        "generated_at": str(pd.Timestamp.now()),
        "summary": {
            "total_points":          len(labeled_df),
            "wrong_way_points":      int((labeled_df["label"] == "wrong_way").sum()),
            "vehicles_total":        int(labeled_df["vehicle_id"].nunique()),
            "vehicles_wrong_way":    int(
                labeled_df.loc[labeled_df["label"] == "wrong_way",
                               "vehicle_id"].nunique()
            ),
            "segments_affected":     int(len(segment_df)),
            "high_risk_segments":    int(
                (segment_df["risk_level"] == "HIGH").sum()
                if not segment_df.empty else 0
            ),
            # NEW
            "collision_risk_count":  len(collision_risks),
            "critical_collisions":   sum(
                1 for r in collision_risks if r["severity"] == "CRITICAL"
            ),
            "danger_zones_count":    len(danger_zones),
        },
        "segment_alerts":  segment_df.to_dict(orient="records") if not segment_df.empty else [],
        "incidents":       incidents,
        "heatmap_points":  heatmap_data,
        # NEW top-level keys
        "danger_zones":    danger_zones,
        "collision_risks": collision_risks,
    }

    os.makedirs(os.path.dirname(alert_path), exist_ok=True)
    with open(alert_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[aggregator] Alerts JSON → {alert_path}")
    print(f"[aggregator] Summary: {output['summary']}")
    return output