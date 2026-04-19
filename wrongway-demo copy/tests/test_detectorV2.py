"""
tests/test_detector_v2.py
--------------------------
Run with: pytest tests/

New tests for v2 features:
  - Speed filter suppression
  - Confidence score formula
  - Segment aggregator risk levels
  - Popup incident builder
"""

import sys
sys.path.append("src")

import pytest
import pandas as pd
import numpy as np

from bearing_utils import angular_difference, is_wrong_way
from detector import (
    apply_noise_filter, compute_confidence,
    LABEL_NORMAL, LABEL_WRONG_WAY, LABEL_CANDIDATE,
    SPEED_FILTER_KMH, WRONG_WAY_ANGLE_THRESHOLD
)
from segment_aggregator import _risk_level, aggregate_segments, build_alert_popup_data


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_labeled_df(vehicle_id, labels, angular_diffs=None,
                    snap_dists=None, run_lengths=None,
                    confidences=None, timestamps=None):
    n = len(labels)
    if timestamps is None:
        timestamps = pd.date_range("2024-01-01 09:00:00", periods=n, freq="2s")
    return pd.DataFrame({
        "vehicle_id":      vehicle_id,
        "lat":             [12.97] * n,
        "lon":             [77.59] * n,
        "timestamp":       timestamps,
        "bearing":         [45.0] * n,
        "label":           labels,
        "angular_diff":    angular_diffs or [0.0] * n,
        "snap_distance_m": snap_dists    or [5.0] * n,
        "run_length":      run_lengths   or [0] * n,
        "confidence":      confidences   or [0.0] * n,
        "edge_id":         ["100_200"] * n,
    })


# ─────────────────────────────────────────────
# Speed filter (tested via label_point indirectly)
# ─────────────────────────────────────────────

class TestSpeedFilter:
    """
    label_point() checks speed_kmh before doing any road snap.
    We test the constant is sane and the threshold is what we expect.
    """
    def test_speed_threshold_constant(self):
        assert SPEED_FILTER_KMH == 15

    def test_angle_threshold_constant(self):
        assert WRONG_WAY_ANGLE_THRESHOLD == 120


# ─────────────────────────────────────────────
# Confidence score
# ─────────────────────────────────────────────

class TestConfidenceScore:

    def _make_ww_df(self, angular_diff, run_length, snap_dist):
        return pd.DataFrame([{
            "vehicle_id":      "v1",
            "label":           LABEL_WRONG_WAY,
            "angular_diff":    angular_diff,
            "run_length":      run_length,
            "snap_distance_m": snap_dist,
        }])

    def test_perfect_confidence(self):
        """180° diff, 10+ run, right on the road → near 100."""
        df = self._make_ww_df(angular_diff=180, run_length=10, snap_dist=0)
        score = compute_confidence(df).iloc[0]
        assert score >= 95, f"Expected ≥95, got {score}"

    def test_minimum_confidence(self):
        """Just above threshold (121°), short run, far snap → low score."""
        df = self._make_ww_df(angular_diff=121, run_length=3, snap_dist=28)
        score = compute_confidence(df).iloc[0]
        assert score < 40, f"Expected <40, got {score}"

    def test_normal_points_get_zero(self):
        """Normal-labeled points always get confidence 0."""
        df = pd.DataFrame([{
            "vehicle_id":      "v1",
            "label":           LABEL_NORMAL,
            "angular_diff":    45.0,
            "run_length":      0,
            "snap_distance_m": 5.0,
        }])
        score = compute_confidence(df).iloc[0]
        assert score == 0

    def test_confidence_always_in_0_100(self):
        """Score must never exceed 100 or go below 0."""
        test_cases = [
            (180, 20, 0),
            (121, 3, 29),
            (150, 5, 10),
            (170, 1, 25),
        ]
        for ang, run, snap in test_cases:
            df = self._make_ww_df(ang, run, snap)
            score = compute_confidence(df).iloc[0]
            assert 0 <= score <= 100, f"Out of range: {score} for ({ang},{run},{snap})"

    def test_higher_angular_diff_means_higher_score(self):
        """More angular deviation = more confident."""
        df_low  = self._make_ww_df(130, 5, 10)
        df_high = self._make_ww_df(175, 5, 10)
        assert compute_confidence(df_high).iloc[0] > compute_confidence(df_low).iloc[0]

    def test_longer_run_means_higher_score(self):
        df_short = self._make_ww_df(160, 3,  10)
        df_long  = self._make_ww_df(160, 10, 10)
        assert compute_confidence(df_long).iloc[0] > compute_confidence(df_short).iloc[0]


# ─────────────────────────────────────────────
# Noise filter (carried over + run_length check)
# ─────────────────────────────────────────────

class TestNoiseFilterV2:

    def _run(self, labels, window=3):
        df = pd.DataFrame({
            "vehicle_id": "v1",
            "raw_label":  labels,
        })
        return apply_noise_filter(df, window=window)

    def test_run_length_populated_on_wrong_way(self):
        labels = ["candidate"] * 5
        result = self._run(labels)
        assert all(result["run_length"] == 5)

    def test_run_length_zero_on_normal(self):
        labels = ["normal"] * 5
        result = self._run(labels)
        assert all(result["run_length"] == 0)

    def test_short_run_run_length_is_zero(self):
        """Run below window size → suppressed → run_length stays 0."""
        labels = ["normal", "candidate", "candidate", "normal"]
        result = self._run(labels, window=3)
        assert all(result["run_length"] == 0)


# ─────────────────────────────────────────────
# Segment aggregator
# ─────────────────────────────────────────────

class TestRiskLevel:

    def test_two_vehicles_is_high(self):
        df = make_labeled_df(["v1", "v2"], [LABEL_WRONG_WAY]*2,
                              confidences=[60, 60])
        assert _risk_level(df) == "HIGH"

    def test_high_confidence_single_vehicle_is_high(self):
        df = make_labeled_df("v1", [LABEL_WRONG_WAY]*3,
                              confidences=[85, 90, 88])
        assert _risk_level(df) == "HIGH"

    def test_medium_confidence_is_medium(self):
        df = make_labeled_df("v1", [LABEL_WRONG_WAY]*3,
                              confidences=[55, 60, 58])
        assert _risk_level(df) == "MEDIUM"

    def test_low_confidence_is_low(self):
        df = make_labeled_df("v1", [LABEL_WRONG_WAY]*3,
                              confidences=[20, 25, 22])
        assert _risk_level(df) == "LOW"


class TestAggregateSegments:

    def test_no_wrong_way_returns_empty(self):
        df = make_labeled_df("v1", ["normal"]*5)
        result = aggregate_segments(df)
        assert result.empty

    def test_aggregation_produces_correct_vehicle_count(self):
        df = pd.concat([
            make_labeled_df("v1", [LABEL_WRONG_WAY]*3, confidences=[70]*3),
            make_labeled_df("v2", [LABEL_WRONG_WAY]*3, confidences=[70]*3),
        ], ignore_index=True)
        result = aggregate_segments(df)
        assert result["vehicle_count"].iloc[0] == 2

    def test_mean_confidence_correct(self):
        df = make_labeled_df("v1", [LABEL_WRONG_WAY]*4, confidences=[60, 80, 70, 90])
        result = aggregate_segments(df)
        assert result["mean_confidence"].iloc[0] == pytest.approx(75.0, abs=0.5)


class TestPopupData:

    def test_incident_splits_on_time_gap(self):
        """Two runs separated by >10s gap → two separate incidents."""
        ts = list(pd.date_range("2024-01-01 09:00:00", periods=3, freq="2s")) + \
             list(pd.date_range("2024-01-01 09:01:00", periods=3, freq="2s"))
        labels = [LABEL_WRONG_WAY] * 6
        df = make_labeled_df("v1", labels, confidences=[70]*6, timestamps=ts)
        incidents = build_alert_popup_data(df)
        assert len(incidents) == 2

    def test_incident_has_required_keys(self):
        df = make_labeled_df("v1", [LABEL_WRONG_WAY]*3, confidences=[70]*3)
        incidents = build_alert_popup_data(df)
        assert len(incidents) == 1
        required = {"vehicle_id", "start_time", "end_time", "duration_seconds",
                    "mean_confidence", "points", "centroid_lat", "centroid_lon"}
        assert required.issubset(set(incidents[0].keys()))