"""
detector.py  (v2 — upgraded)
-----------------------------
Person A — Core Detection Pipeline

Changes from v1:
  + Confidence score (0–100) on every wrong_way point
  + Speed-aware false positive suppression
  + Snap distance stored per point (fed into confidence)
  + Run-length stored per point (for segment aggregator + confidence)
  + label_point now returns snap_distance so scorer can use it
"""

import pandas as pd
import numpy as np
import networkx as nx

from bearing_utils import angular_difference, is_wrong_way
from road_graph import snap_to_road

# ─────────────────────────────────────────────
# Tuneable constants
# ─────────────────────────────────────────────

WRONG_WAY_ANGLE_THRESHOLD = 120   # degrees — default; overridden per-run via angle_threshold param
PERSISTENCE_WINDOW        = 3     # consecutive flagged points to confirm
SPEED_FILTER_KMH          = 15    # suppress wrong-way if speed below this (parking/u-turn)

LABEL_NORMAL    = "normal"
LABEL_CANDIDATE = "candidate"   # internal only
LABEL_WRONG_WAY = "wrong_way"


# ─────────────────────────────────────────────
# Step 1: Per-point snap + raw label
# ─────────────────────────────────────────────

def label_point(row: pd.Series, G: nx.MultiDiGraph,
                angle_threshold: int = WRONG_WAY_ANGLE_THRESHOLD) -> dict:
    """
    Snap a GPS point to its nearest road, compare bearings, emit raw label.

    NEW in v2:
      - Returns snap_distance_m (used by confidence scorer)
      - Returns edge_id tuple (used by segment aggregator)
      - Checks speed column if present — suppresses flag when speed < threshold

    NEW in v3:
      - angle_threshold: override the module-level WRONG_WAY_ANGLE_THRESHOLD
        per call.  Passed through from run_detection() so multi-threshold
        runs stay self-consistent without mutating global state.
    """
    # Speed gate — if vehicle is barely moving, don't flag
    speed = row.get("speed_kmh", None)
    if speed is not None and not pd.isna(speed) and speed < SPEED_FILTER_KMH:
        return {
            "road_bearing":    None,
            "angular_diff":    None,
            "snap_distance_m": None,
            "edge_id":         None,
            "raw_label":       LABEL_NORMAL,
            "suppressed_by":   "speed_filter",
        }

    snap = snap_to_road(G, row["lat"], row["lon"])

    if snap is None:
        return {
            "road_bearing":    None,
            "angular_diff":    None,
            "snap_distance_m": None,
            "edge_id":         None,
            "raw_label":       LABEL_NORMAL,
            "suppressed_by":   "no_snap",
        }

    u, v, key, road_bearing, snap_dist_m = snap   # snap_to_road v2 returns 5-tuple
    diff = angular_difference(row["bearing"], road_bearing)
    raw_label = LABEL_CANDIDATE if diff > angle_threshold else LABEL_NORMAL

    return {
        "road_bearing":    round(road_bearing, 2),
        "angular_diff":    round(diff, 2),
        "snap_distance_m": round(snap_dist_m, 2),
        "edge_id":         f"{u}_{v}",          # string key for segment aggregator
        "raw_label":       raw_label,
        "suppressed_by":   None,
    }


# ─────────────────────────────────────────────
# Step 2: Noise filter + run-length tracking
# ─────────────────────────────────────────────

def apply_noise_filter(df: pd.DataFrame,
                       window: int = PERSISTENCE_WINDOW) -> pd.DataFrame:
    """
    Confirms wrong-way events from candidate runs.

    NEW in v2:
      - Also returns run_length column (how many consecutive points in the run)
        This feeds into the confidence scorer.

    Returns:
        DataFrame with two new columns: label, run_length
    """
    final_labels  = df["raw_label"].copy()
    run_lengths   = pd.Series(0, index=df.index)

    for vehicle_id, group in df.groupby("vehicle_id", sort=False):
        indices = group.index.tolist()
        raw     = group["raw_label"].tolist()
        confirmed   = [LABEL_NORMAL] * len(raw)
        run_len_arr = [0] * len(raw)

        i = 0
        while i < len(raw):
            if raw[i] == LABEL_CANDIDATE:
                run_start = i
                while i < len(raw) and raw[i] == LABEL_CANDIDATE:
                    i += 1
                run_end = i
                run_len = run_end - run_start

                if run_len >= window:
                    for j in range(run_start, run_end):
                        confirmed[j]   = LABEL_WRONG_WAY
                        run_len_arr[j] = run_len
            else:
                i += 1

        for idx, label, rlen in zip(indices, confirmed, run_len_arr):
            final_labels.at[idx] = label
            run_lengths.at[idx]  = rlen

    result = df.copy()
    result["label"]      = final_labels
    result["run_length"] = run_lengths
    return result


# ─────────────────────────────────────────────
# Step 3: Confidence scoring
# ─────────────────────────────────────────────

def compute_confidence(df: pd.DataFrame,
                       angle_threshold: int = WRONG_WAY_ANGLE_THRESHOLD) -> pd.Series:
    """
    Assigns a 0–100 confidence score to every wrong_way point.
    Normal points get score 0.

    Formula (three components, weighted):
      A) Angular component  (50 pts): how far past angle_threshold the bearing diff is
            score_A = (angular_diff - angle_threshold) / (180 - angle_threshold) * 50
                      → 0 at exactly the threshold, max 50 at 180°
      B) Persistence component (30 pts): how long the run is
            score_B = min(run_length / 10, 1) * 30     → saturates at 10 pts
      C) Snap quality component (20 pts): how close to the road centreline
            score_C = max(0, 1 - snap_distance_m / 30) * 20  → max 20 if <1m

    All three are clamped to their max before summing.
    Final score clamped to [0, 100].

    Why these weights?
      Angular diff is the primary signal — it gets the most weight.
      Persistence confirms it's not a glitch — second weight.
      Snap quality is a trust modifier — lowest weight.

    NEW in v3:
      angle_threshold is now a parameter so that score_A is always
      normalised relative to the threshold used for flagging.  At 90°
      the headroom is 90°; at 150° it is 30° — the scale adjusts
      automatically so confidence values are comparable across runs.
    """
    scores = pd.Series(0.0, index=df.index)

    ww_mask = df["label"] == LABEL_WRONG_WAY
    if ww_mask.sum() == 0:
        return scores

    ww = df[ww_mask].copy()

    # Component A — angular: normalised over the headroom above the threshold
    headroom = max(180 - angle_threshold, 1)   # degrees between threshold and 180°
    score_A = ((ww["angular_diff"] - angle_threshold) / headroom * 50).clip(0, 50)

    # Component B — persistence
    score_B = (ww["run_length"] / 10).clip(0, 1) * 30

    # Component C — snap quality (closer = better)
    snap_ok = ww["snap_distance_m"].notna()
    score_C = pd.Series(0.0, index=ww.index)
    score_C[snap_ok] = (1 - ww.loc[snap_ok, "snap_distance_m"] / 30).clip(0, 1) * 20

    total = (score_A + score_B + score_C).clip(0, 100).round(1)
    scores.loc[ww_mask] = total

    return scores


# ─────────────────────────────────────────────
# Step 4: Full pipeline
# ─────────────────────────────────────────────

def run_detection(input_df: pd.DataFrame, G: nx.MultiDiGraph,
                  angle_threshold: int = WRONG_WAY_ANGLE_THRESHOLD) -> pd.DataFrame:
    """
    Full v3 detection pipeline.

    Input columns:  vehicle_id, lat, lon, timestamp, bearing
    Optional input: speed_kmh  (Person B adds this — suppresses false positives)

    Output adds:
      road_bearing, angular_diff, snap_distance_m, edge_id,
      label, run_length, confidence, suppressed_by

    NEW in v3:
      angle_threshold — override the flagging angle per run.
        90°  : most sensitive, catches subtle wrong-way entries
        120° : default, current behaviour
        150° : most conservative, near-head-on conflicts only
      Passed through to label_point() and compute_confidence() so that
      both the flagging logic and the confidence formula stay consistent.
    """
    print(f"[detector] Processing {len(input_df)} GPS points "
          f"(angle_threshold={angle_threshold}°)...")

    # Step 1: Per-point snap + raw label
    results = input_df.apply(
        lambda row: label_point(row, G, angle_threshold=angle_threshold),
        axis=1, result_type="expand"
    )
    df = pd.concat([input_df.reset_index(drop=True), results], axis=1)

    # Step 2: Noise filter + run lengths
    df = apply_noise_filter(df)

    # Step 3: Confidence scores (normalised to the same threshold used for flagging)
    df["confidence"] = compute_confidence(df, angle_threshold=angle_threshold)

    # Drop internal column
    df.drop(columns=["raw_label"], inplace=True, errors="ignore")

    n_wrong = (df["label"] == LABEL_WRONG_WAY).sum()
    n_total = len(df)
    print(f"[detector] Done. {n_wrong}/{n_total} points labeled wrong_way.")
    print(f"[detector] Mean confidence on wrong_way points: "
          f"{df.loc[df['label']==LABEL_WRONG_WAY, 'confidence'].mean():.1f}")

    return df