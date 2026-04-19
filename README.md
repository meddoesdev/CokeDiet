# Wrong-Way Vehicle Detection System

**MMCAI-03 · Manipal Institute of Technology, Bengaluru**
**Harman Centre of Excellence in Autonomous Mobility — Hackathon 2024**
**Problem Statement 3: Wrong-Way Driver Detection Beyond Ramps**

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [How It Works](#how-it-works)
3. [Repository Structure](#repository-structure)
4. [Quick Start](#quick-start)
5. [Pipeline Reference](#pipeline-reference)
6. [Output Files](#output-files)
7. [Features](#features)
8. [Configuration](#configuration)
9. [Running Tests](#running-tests)
10. [Design Decisions](#design-decisions)
11. [Known Limitations](#known-limitations)
12. [Assumptions](#assumptions)

---

## What This Is

A **GPS-only, rule-based wrong-way vehicle detection system** that identifies vehicles travelling against the allowed direction on one-way roads using heading comparison against OpenStreetMap road geometry.

No machine learning. No cameras. No proprietary data. Every detection is geometrically explainable.

The system outputs a 0–100 confidence score per detection, aggregates incidents to road-segment level with risk classification (HIGH / MEDIUM / LOW), predicts danger zones via dead reckoning, and detects oncoming collision risks between vehicle pairs. All outputs feed a live animated Leaflet.js dashboard.

---

## How It Works

### Core algorithm (plain English)

```
For each GPS point in the trace:
  1. Snap to nearest OSM road edge (within 30m)
  2. Compute angular_difference(vehicle_bearing, road_bearing)
  3. If angular_diff > 120°  →  mark as CANDIDATE
  4. If speed_kmh < 15       →  suppress regardless (speed filter)
  5. If 3+ consecutive CANDIDATE points  →  confirm as WRONG_WAY
  6. Assign confidence score 0–100 based on:
       angular deviation  (50 pts max)
       run length         (30 pts max)
       snap quality       (20 pts max)
```

### Why rule-based and not ML?

- **Explainability**: every alert cites exact bearing values, angular difference, and point count
- **Zero training data**: works on any OSM-covered city from day one
- **Fast**: full pipeline runs in under 15 seconds on 200 GPS points
- **Auditable**: regulators and OEMs can inspect exactly why a vehicle was flagged

### Pipeline overview

```
gps_traces.csv
      │
      ▼
  road_graph.py          ← download + cache OSM graph, add edge bearings
      │
      ▼
  detector.py            ← snap → compare → filter → confidence score
      │
      ▼
  segment_aggregator.py  ← aggregate to segments, build popup data
      │
      ▼
  danger_zone.py         ← dead reckoning + collision risk detection
      │
      ▼
  labeled_traces.csv  +  segment_alerts.json
      │
      ▼
  inject_data.py         ← embed data into map.html
      │
      ▼
  map.html               ← open in browser, press Play
```

---

## Repository Structure

```
wrong-way-detection/
│
├── README.md                      ← this file
├── requirements.txt               ← pip dependencies
├── conftest.py                    ← pytest path config (run from root)
│
├── src/
│   ├── bearing_utils.py           ← pure bearing/angle math (no dependencies)
│   ├── road_graph.py              ← OSM graph download, caching, snap_to_road()
│   ├── detector.py                ← core detection pipeline
│   ├── segment_aggregator.py      ← segment-level aggregation + JSON export
│   ├── danger_zone.py             ← danger zone predictor + collision detector
│   ├── run_detection.py           ← entry point — run this to process traces
│   └── simulate_traces.py         ← GPS trace simulator (Person B)
│
├── data/
│   ├── raw/
│   │   ├── gps_traces.csv         ← input: simulated GPS traces
│   │   └── road_graph.graphml     ← cached OSM graph (auto-generated)
│   └── output/
│       ├── labeled_traces.csv     ← detection output with confidence scores
│       └── segment_alerts.json    ← aggregated alerts, incidents, heatmap data
│
├── map.html                       ← interactive demo (open in browser)
├── inject_data.py                 ← injects CSV + JSON into map.html
│
├── tests/
│   ├── __init__.py
│   ├── test_detector.py           ← unit tests: bearing math, noise filter
│   └── test_danger_zone.py        ← unit tests: dead reckoning, collision logic
│
└── docs/
    └── logic_explained.md         ← detailed writeup of detection logic + false positives
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate GPS traces

```bash
cd src
python simulate_traces.py
```

This downloads the Connaught Place, New Delhi road graph from OSM (first run only, ~30–60s), then generates `data/raw/gps_traces.csv` with four vehicles:

| Vehicle | Scenario |
|---|---|
| `veh_001` | Normal vehicle — follows road directions correctly |
| `veh_wrongway` | Wrong-way intruder — drives against one-way traffic |
| `veh_uturn` | U-turn false positive — slow speed suppresses the flag |
| `veh_003` | Second wrong-way vehicle — different route, enables risk classification |

### 3. Run the detection pipeline

```bash
python run_detection.py
```

Expected output:
```
==================================================
  Wrong-Way Detection Pipeline v3
==================================================

[run_detection] 450 rows, 4 vehicles
[road_graph] Loading cached graph from data/raw/road_graph.graphml
[detector] Processing 450 GPS points...
[detector] Done. 82/450 points labeled wrong_way.
[detector] Mean confidence on wrong_way points: 91.4
[danger_zone] Predicted danger zones for 3 wrong-way vehicle(s).
[danger_zone] ww-vs-normal: 3 event(s)
[danger_zone] ww-vs-ww:     1 event(s)
[danger_zone] Total: 4 collision risk(s) — CRITICAL: 2, HIGH: 2
[aggregator] Labeled CSV → data/output/labeled_traces.csv
[aggregator] Alerts JSON → data/output/segment_alerts.json
```

### 4. Inject data into the map

```bash
cd ..          # back to project root
python inject_data.py
```

### 5. Open the demo

Open `map.html` in any modern browser. Press **PLAY**.

---

## Pipeline Reference

### `src/bearing_utils.py`

Pure trigonometry — no OSM or pandas dependencies. Safe to unit test in isolation.

| Function | What it does |
|---|---|
| `compute_bearing(lat1, lon1, lat2, lon2)` | Compass bearing (0–360°) between two GPS points |
| `angular_difference(bearing_a, bearing_b)` | Smallest angle between two bearings, handles 0/360 wraparound |
| `is_wrong_way(vehicle_bearing, road_bearing, threshold=120)` | Returns True if angular diff exceeds threshold |
| `reverse_bearing(bearing)` | Returns the opposite direction |

### `src/road_graph.py`

Downloads and caches the OSM road graph. Core function:

```python
snap = snap_to_road(G, lat, lon)
# Returns: (u, v, key, road_bearing, snap_distance_m)
# Returns None if no road within MAX_SNAP_DISTANCE_M (30m)
```

The graph is saved to `data/raw/road_graph.graphml` on first download and reloaded from disk on subsequent runs. Never re-downloads unless you delete the file.

### `src/detector.py`

The main detection logic. Three stages:

**Stage 1 — `label_point(row, G)`**
Snaps each GPS point to the nearest road edge. Applies speed gate first (returns `normal` if `speed_kmh < 15`). Computes angular difference. Returns `candidate` or `normal` with snap metadata.

**Stage 2 — `apply_noise_filter(df)`**
Scans each vehicle's point sequence for runs of consecutive `candidate` labels. Runs ≥ `PERSISTENCE_WINDOW` (default: 3) become `wrong_way`. Shorter runs revert to `normal`. Also records `run_length` per confirmed point.

**Stage 3 — `compute_confidence(df)`**
Assigns a 0–100 confidence score to every `wrong_way` point:
- Angular component (50 pts): `(angular_diff - 120) / 60 * 50`
- Persistence component (30 pts): `min(run_length / 10, 1) * 30`
- Snap quality component (20 pts): `(1 - snap_distance_m / 30) * 20`

### `src/segment_aggregator.py`

Groups `wrong_way` points by `(edge_id, time_window)` into segment-level alerts. Classifies each segment as HIGH / MEDIUM / LOW:

- **HIGH**: 2+ distinct vehicles triggered it, OR mean confidence ≥ 80
- **MEDIUM**: 1 vehicle, confidence 50–79
- **LOW**: 1 vehicle, confidence < 50

Exports two files consumed by the map UI:
- `labeled_traces.csv` — full point data for animated replay
- `segment_alerts.json` — aggregated segments, incident records, heatmap weights, danger zones, collision risks

### `src/danger_zone.py`

Two predictive safety features:

**Danger zone predictor — `predict_danger_zones(df)`**
Dead reckoning from the last confirmed wrong-way position. Projects the next `PROJECTION_STEPS` (default: 5) GPS positions forward using:
```
delta_lat = (speed_ms * interval_s * cos(bearing_rad)) / 111_000
delta_lon = (speed_ms * interval_s * sin(bearing_rad)) / (111_000 * cos(lat_rad))
```
Output: `predicted_path` array per vehicle. Person B renders this as an orange dashed polyline.

**Collision risk detector — `detect_collision_risks(df)`**
Runs two sub-detectors:

1. **ww-vs-normal**: wrong-way vehicle vs normal vehicle on the same edge, within 4 seconds, within 80 metres, opposing bearings. Severity: CRITICAL (<40m) or HIGH (40–80m).

2. **ww-vs-ww**: any two wrong-way vehicles converging on each other within 80 metres. Does not require same edge or opposing bearings — uses bearing-to-target check to confirm convergence. Always CRITICAL.

---

## Output Files

### `data/output/labeled_traces.csv`

Every input GPS row plus these added columns:

| Column | Description |
|---|---|
| `label` | `normal` or `wrong_way` |
| `confidence` | 0–100 score (0 for all normal points) |
| `road_bearing` | OSM allowed direction at this point (degrees) |
| `angular_diff` | Degrees between vehicle and road bearing |
| `snap_distance_m` | Metres from GPS point to road centreline |
| `run_length` | Consecutive wrong_way points in this confirmed run |
| `edge_id` | OSM edge identifier (`u_v` format) |
| `suppressed_by` | `null`, `speed_filter`, or `no_snap` |

### `data/output/segment_alerts.json`

Top-level keys:

```json
{
  "generated_at": "...",
  "summary": {
    "total_points": 450,
    "wrong_way_points": 82,
    "vehicles_total": 4,
    "vehicles_wrong_way": 3,
    "segments_affected": 12,
    "high_risk_segments": 11,
    "collision_risk_count": 4,
    "critical_collisions": 2,
    "danger_zones_count": 3
  },
  "segment_alerts": [...],
  "incidents": [...],
  "heatmap_points": [...],
  "danger_zones": [...],
  "collision_risks": [...]
}
```

---

## Features

| Feature | Description |
|---|---|
| **Bearing comparison** | Vehicle heading vs OSM road bearing, 120° threshold |
| **Persistence filter** | 3 consecutive flagged points required — eliminates GPS noise |
| **Speed filter** | Suppresses flags when speed < 15 km/h — eliminates U-turns |
| **Confidence scores** | 0–100 per detection, decomposed into 3 geometric components |
| **Segment risk levels** | HIGH / MEDIUM / LOW per road segment across time windows |
| **Danger zone predictor** | Dead reckoning: projects next 10 seconds of vehicle trajectory |
| **Collision risk detector** | ww-vs-normal and ww-vs-ww proximity detection |
| **Live animated replay** | Play/pause/reset with per-timestep alert banners and sounds |
| **Heatmap overlay** | Confidence-weighted intensity per wrong_way GPS point |
| **Threshold tuning panel** | Live sliders: 90° aggressive / 120° optimal / 150° conservative |
| **Explainability popup** | Click any alert marker to see exact bearing math and component scores |
| **Speed filter toggle** | Show/hide suppressed points — demonstrates false positive handling |

---

## Configuration

All tunable constants are at the top of their respective files:

**`src/detector.py`**
```python
WRONG_WAY_ANGLE_THRESHOLD = 120   # degrees — flag if angular diff exceeds this
PERSISTENCE_WINDOW        = 3     # consecutive flagged points to confirm
SPEED_FILTER_KMH          = 15    # suppress if speed below this
```

**`src/road_graph.py`**
```python
PLACE_NAME           = "Connaught Place, New Delhi, India"
GRAPH_SAVE_PATH      = "data/raw/road_graph.graphml"
MAX_SNAP_DISTANCE_M  = 30
```

**`src/danger_zone.py`**
```python
PROJECTION_STEPS      = 5    # future positions to predict
PROJECTION_INTERVAL_S = 2    # seconds between projected points
COLLISION_THRESHOLD_M = 80   # proximity threshold for collision flag
TIME_WINDOW_S         = 4    # max timestamp gap to count as "same moment"
```

To change the city, update `PLACE_NAME` in `road_graph.py` and delete `data/raw/road_graph.graphml` to trigger a fresh download.

---

## Running Tests

```bash
# From project root
pytest tests/ -v
```

Tests are pure unit tests — no OSM network calls, no file I/O. All 34 tests pass in under 2 seconds.

**`tests/test_detector.py`** — covers:
- `angular_difference()` wraparound (5° vs 355° = 10°, not 350°)
- `is_wrong_way()` boundary conditions
- `compute_bearing()` cardinal directions
- `apply_noise_filter()` run detection, multi-vehicle isolation
- Confidence score formula: monotonicity, bounds, component ordering

**`tests/test_danger_zone.py`** — covers:
- `haversine_m()` accuracy and symmetry
- `_project_position()` direction, distance scaling, precision
- `predict_danger_zones()` step count, path direction, multi-vehicle
- `_are_heading_toward_each_other()` head-on, perpendicular, wraparound
- `detect_collision_risks()` all four conditions, deduplication, severity thresholds

---

## Design Decisions

**Why 120° as the threshold?**
At 90° you flag vehicles crossing perpendicularly at intersections. At 180° you only catch perfectly head-on cases. 120° gives a ±60° tolerance around the opposing direction — large enough to ignore normal road curvature and GPS jitter, tight enough to catch deliberate wrong-way driving.

**Why 3 points for the persistence window?**
At a 2-second GPS sampling rate, 3 points = 6 seconds of sustained wrong-way travel. Single-point glitches (GPS multipath, sharp turns) produce at most 1–2 consecutive candidates. 3 is the minimum that reliably separates signal from noise.

**Why 15 km/h for the speed cutoff?**
U-turns, parking maneuvers, and slow intersection navigation all produce temporary wrong-way bearings. 15 km/h is fast enough to be deliberate driving, slow enough to filter everything that isn't.

**Why not map matching (HMM)?**
Hidden Markov Model map matching is more accurate but adds significant complexity. For a hackathon prototype demonstrating the detection concept, nearest-edge snapping with a 30m distance cutoff is sufficient. HMM is the recommended next step for production.

**Why export `edge_id` as a string?**
OSM node IDs are large integers that exceed JavaScript's safe integer range. Storing them as `"u_v"` strings avoids precision loss when the JSON is parsed in the browser.

---

## Known Limitations

**Straight-line GPS interpolation**
The `simulate_traces.py` simulator interpolates linearly between OSM node coordinates. Real GPS traces follow actual road curves. This causes the ghost path on the map to show diagonal shortcuts across curves — not a detection logic problem, but a simulation fidelity issue. Use real GPS data or OSM edge geometry sampling for production.

**Dead reckoning drift**
The danger zone predictor projects positions in a straight line using constant bearing and speed. Real vehicles follow road curves. The orange predicted path visually diverges from the road after ~2–3 steps. Production fix: extend the projection along the snapped edge's geometry rather than free-form lat/lon arithmetic.

**No real-time streaming**
The pipeline is batch — it reads a complete CSV and processes it offline. Real-time deployment requires a streaming architecture (e.g., MQTT ingest → per-vehicle state machine → rolling window detection).

**OSM data lag**
If a one-way road has been temporarily reversed for roadworks and OSM hasn't been updated, the detector will flag legitimate vehicles. Mitigation: ingest real-time roadworks feeds and suppress alerts on affected segments.

**Divided highways**
GPS imprecision on wide dual carriageways can snap a vehicle to the opposing carriageway's OSM edge. Mitigation: increase snap distance tolerance or use lane-level map data.

---

## Assumptions

1. All monitored vehicles have active GPS transmitting `lat`, `lon`, `bearing`, and `speed_kmh` at ≥ 0.5 Hz
2. Vehicles or roadside units can relay GPS payloads to the detection server with under 5 seconds latency
3. OpenStreetMap road direction data is current and accurate for the target area
4. Road segments are tagged as one-way in OSM — bidirectional roads are not detected as wrong-way
5. GPS traces in this prototype are synthetically generated from OSM road geometry — no live vehicle data is used
6. Vehicle speed is available in the GPS payload — if absent, the speed filter is skipped (pipeline warns)
7. System is validated on a 1 km² urban grid (Connaught Place, New Delhi) — highway ramp geometry is out of scope
8. The OSM graph covers the operational area — remote or unmapped roads cannot be used for detection

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Road graph | `osmnx` 1.9.3 |
| Data processing | `pandas` 2.2.2, `numpy` 1.26.4 |
| Geometry | `shapely` 2.0.4 |
| Map visualisation | Leaflet.js 1.9.4, leaflet.heat 0.2.0 |
| GPS simulation | `osmnx` shortest path + custom densification |
| Testing | `pytest` 8.2.2 |
| Data injection | Python `re`, `json` |

---

## Licence

Built for the Harman Centre of Excellence in Autonomous Mobility Hackathon 2024. For academic and demonstration purposes only.
