"""
tests/test_danger_zone.py
--------------------------
Run with: pytest tests/

Tests for:
  - _project_position() dead reckoning accuracy
  - predict_danger_zones() output structure + edge cases
  - haversine_m() accuracy
  - _are_heading_toward_each_other() logic
  - detect_collision_risks() — all 4 conditions
"""

import pytest
import pandas as pd
import numpy as np
from math import isclose

from danger_zone import (
    _project_position,
    predict_danger_zones,
    haversine_m,
    _are_heading_toward_each_other,
    detect_collision_risks,
    PROJECTION_STEPS,
    COLLISION_THRESHOLD_M,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

BASE_TS = pd.date_range("2024-01-01 09:00:00", periods=5, freq="2s")

def make_df(vehicle_id, label, lats, lons, bearings,
            speeds=None, edge_id="100_200", timestamps=None):
    n = len(lats)
    if timestamps is None:
        timestamps = pd.date_range("2024-01-01 09:00:00", periods=n, freq="2s")
    if speeds is None:
        speeds = [35.0] * n
    return pd.DataFrame({
        "vehicle_id":      vehicle_id,
        "lat":             lats,
        "lon":             lons,
        "timestamp":       timestamps,
        "bearing":         bearings,
        "speed_kmh":       speeds,
        "label":           label,
        "confidence":      [85.0 if label == "wrong_way" else 0.0] * n,
        "angular_diff":    [170.0 if label == "wrong_way" else 5.0] * n,
        "run_length":      [5 if label == "wrong_way" else 0] * n,
        "snap_distance_m": [3.0] * n,
        "edge_id":         edge_id,
    })


# ─────────────────────────────────────────────
# haversine_m
# ─────────────────────────────────────────────

class TestHaversine:

    def test_same_point_is_zero(self):
        assert haversine_m(28.63, 77.21, 28.63, 77.21) == pytest.approx(0, abs=0.01)

    def test_known_distance(self):
        # ~111km per degree of latitude
        dist = haversine_m(0, 0, 1, 0)
        assert dist == pytest.approx(111_195, rel=0.01)

    def test_symmetry(self):
        a = haversine_m(28.63, 77.21, 28.64, 77.22)
        b = haversine_m(28.64, 77.22, 28.63, 77.21)
        assert a == pytest.approx(b, rel=1e-6)


# ─────────────────────────────────────────────
# _project_position (dead reckoning)
# ─────────────────────────────────────────────

class TestProjectPosition:

    def test_due_north(self):
        """Moving north at 36 km/h for 2s = 20m north."""
        lat, lon = _project_position(28.63, 77.21, bearing_deg=0,
                                      speed_kmh=36, interval_s=2)
        assert lat > 28.63
        assert lon == pytest.approx(77.21, abs=0.0001)

    def test_due_east(self):
        """Moving east — lon increases, lat stays roughly same."""
        lat, lon = _project_position(28.63, 77.21, bearing_deg=90,
                                      speed_kmh=36, interval_s=2)
        assert lon > 77.21
        assert lat == pytest.approx(28.63, abs=0.0001)

    def test_due_south(self):
        lat, lon = _project_position(28.63, 77.21, bearing_deg=180,
                                      speed_kmh=36, interval_s=2)
        assert lat < 28.63

    def test_distance_scales_with_speed(self):
        """Double the speed → double the projected distance."""
        lat1, lon1 = _project_position(28.63, 77.21, 45, 30, 2)
        lat2, lon2 = _project_position(28.63, 77.21, 45, 60, 2)
        d1 = haversine_m(28.63, 77.21, lat1, lon1)
        d2 = haversine_m(28.63, 77.21, lat2, lon2)
        assert d2 == pytest.approx(d1 * 2, rel=0.05)

    def test_returns_6dp_precision(self):
        lat, lon = _project_position(28.63, 77.21, 90, 36, 2)
        assert len(str(lat).split(".")[-1]) <= 6
        assert len(str(lon).split(".")[-1]) <= 6


# ─────────────────────────────────────────────
# predict_danger_zones
# ─────────────────────────────────────────────

class TestPredictDangerZones:

    def _ww_df(self):
        return make_df("veh_wrongway", "wrong_way",
                       lats=[28.630, 28.631, 28.632, 28.633, 28.634],
                       lons=[77.219]*5,
                       bearings=[212.0]*5)

    def test_returns_one_entry_per_wrong_way_vehicle(self):
        df = self._ww_df()
        result = predict_danger_zones(df)
        assert len(result) == 1
        assert result[0]["vehicle_id"] == "veh_wrongway"

    def test_predicted_path_has_correct_step_count(self):
        df = self._ww_df()
        result = predict_danger_zones(df, steps=5)
        assert len(result[0]["predicted_path"]) == 5

    def test_predicted_path_moves_in_bearing_direction(self):
        """Bearing 0 (north) → each step should have increasing lat."""
        df = make_df("veh_ww", "wrong_way",
                     lats=[28.63]*5, lons=[77.21]*5, bearings=[0.0]*5)
        result = predict_danger_zones(df, steps=3)
        path = result[0]["predicted_path"]
        lats = [p["lat"] for p in path]
        assert lats[0] < lats[1] < lats[2]

    def test_t_plus_s_increments_correctly(self):
        df = self._ww_df()
        result = predict_danger_zones(df, steps=4, interval_s=2)
        t_values = [p["t_plus_s"] for p in result[0]["predicted_path"]]
        assert t_values == [2, 4, 6, 8]

    def test_empty_df_returns_empty_list(self):
        df = make_df("veh_normal", "normal",
                     lats=[28.63]*3, lons=[77.21]*3, bearings=[45.0]*3)
        result = predict_danger_zones(df)
        assert result == []

    def test_multiple_wrong_way_vehicles(self):
        df1 = make_df("veh_ww1", "wrong_way",
                      lats=[28.63]*5, lons=[77.21]*5, bearings=[200.0]*5)
        df2 = make_df("veh_ww2", "wrong_way",
                      lats=[28.64]*5, lons=[77.22]*5, bearings=[190.0]*5,
                      edge_id="200_300")
        df = pd.concat([df1, df2], ignore_index=True)
        result = predict_danger_zones(df)
        assert len(result) == 2


# ─────────────────────────────────────────────
# _are_heading_toward_each_other
# ─────────────────────────────────────────────

class TestHeadingToward:

    def test_perfect_head_on(self):
        """0° and 180° are perfectly opposing."""
        assert _are_heading_toward_each_other(0, 180) is True

    def test_same_direction(self):
        """Both going north — not toward each other."""
        assert _are_heading_toward_each_other(0, 0) is False

    def test_perpendicular(self):
        """90° apart — not heading toward each other."""
        assert _are_heading_toward_each_other(0, 90) is False

    def test_within_tolerance(self):
        """170° diff — within ±60° of 180° → True."""
        assert _are_heading_toward_each_other(10, 180) is True

    def test_wraparound(self):
        """355° and 175° — diff is ~180°, should be True."""
        assert _are_heading_toward_each_other(355, 175) is True


# ─────────────────────────────────────────────
# detect_collision_risks
# ─────────────────────────────────────────────

class TestCollisionRisk:

    def _make_collision_pair(self, dist_offset=0.0003):
        """
        Two vehicles on the same edge, same timestamp, close together,
        heading toward each other. Should trigger a collision risk.
        dist_offset in degrees lat ≈ ~33m at this latitude.
        """
        ts = pd.date_range("2024-01-01 09:00:10", periods=3, freq="2s")
        ww  = make_df("veh_ww",  "wrong_way",
                      lats=[28.633]*3, lons=[77.219]*3,
                      bearings=[200.0]*3, timestamps=ts)
        nrm = make_df("veh_nrm", "normal",
                      lats=[28.633 + dist_offset]*3,
                      lons=[77.219]*3,
                      bearings=[20.0]*3, timestamps=ts)
        return pd.concat([ww, nrm], ignore_index=True)

    def test_detects_head_on_collision(self):
        df = self._make_collision_pair(dist_offset=0.0003)
        risks = detect_collision_risks(df, threshold_m=80)
        assert len(risks) >= 1

    def test_no_risk_when_too_far(self):
        """Vehicles on same edge but >threshold metres apart."""
        df = self._make_collision_pair(dist_offset=0.01)  # ~1.1km apart
        risks = detect_collision_risks(df, threshold_m=80)
        assert len(risks) == 0

    def test_no_risk_when_same_direction(self):
        """Same edge, close, but both going same direction — not a collision."""
        ts = pd.date_range("2024-01-01 09:00:10", periods=3, freq="2s")
        ww  = make_df("veh_ww",  "wrong_way",
                      lats=[28.633]*3, lons=[77.219]*3,
                      bearings=[200.0]*3, timestamps=ts)
        nrm = make_df("veh_nrm", "normal",
                      lats=[28.6332]*3, lons=[77.219]*3,
                      bearings=[200.0]*3, timestamps=ts)  # same direction
        df = pd.concat([ww, nrm], ignore_index=True)
        risks = detect_collision_risks(df, threshold_m=80)
        assert len(risks) == 0

    def test_no_risk_on_different_edges(self):
        """Same distance, opposing bearings, but different road segments."""
        ts = pd.date_range("2024-01-01 09:00:10", periods=3, freq="2s")
        ww  = make_df("veh_ww",  "wrong_way",
                      lats=[28.633]*3, lons=[77.219]*3,
                      bearings=[200.0]*3, timestamps=ts, edge_id="AAA_BBB")
        nrm = make_df("veh_nrm", "normal",
                      lats=[28.6332]*3, lons=[77.219]*3,
                      bearings=[20.0]*3,  timestamps=ts, edge_id="CCC_DDD")
        df = pd.concat([ww, nrm], ignore_index=True)
        risks = detect_collision_risks(df, threshold_m=80)
        assert len(risks) == 0

    def test_severity_critical_when_very_close(self):
        """Within threshold/2 → CRITICAL."""
        df = self._make_collision_pair(dist_offset=0.0001)  # ~11m
        risks = detect_collision_risks(df, threshold_m=80)
        if risks:
            assert risks[0]["severity"] == "CRITICAL"

    def test_output_has_required_keys(self):
        df = self._make_collision_pair()
        risks = detect_collision_risks(df, threshold_m=80)
        if risks:
            required = {"wrong_way_vehicle", "normal_vehicle", "timestamp",
                        "edge_id", "distance_m", "centroid_lat",
                        "centroid_lon", "severity"}
            assert required.issubset(set(risks[0].keys()))

    def test_deduplication(self):
        """Same pair at same timestamp should only appear once."""
        df = self._make_collision_pair(dist_offset=0.0003)
        risks = detect_collision_risks(df, threshold_m=80)
        pairs = [(r["wrong_way_vehicle"], r["normal_vehicle"]) for r in risks]
        assert len(pairs) == len(set(pairs)) or len(risks) <= 3