# Wrong-Way Detection: Logic Explained

> **Author:** Person A
> **Fill this in during Hours 11–14 while Person B builds the map.**

---

## 1. How Does Bearing Comparison Work?

### What is a bearing?
A bearing is a compass direction expressed as degrees clockwise from north (0°–360°).
- 0° = North
- 90° = East
- 180° = South
- 270° = West

Every road segment in OpenStreetMap has a bearing derived from its geometry — the direction you'd travel if you walked from node A to node B along that edge.

### How do we get the vehicle's bearing?
Person B's GPS trace includes a `bearing` column. Each row's bearing is the direction from the previous GPS point to the current one, computed using the **Haversine formula** on consecutive (lat, lon) pairs.

### How is angular difference calculated?
We compare the vehicle's heading to the road's allowed direction:

```
diff = |vehicle_bearing - road_bearing| mod 360
angular_difference = min(diff, 360 - diff)
```

The `min(diff, 360 - diff)` handles **wraparound** — e.g. a bearing of 5° and 355° are only 10° apart in reality, not 350°.

### Why 120° as the threshold?
- A vehicle going the **same direction** as the road: ~0° difference
- A vehicle going **perpendicular** (e.g. crossing): ~90° difference
- A vehicle going **head-on against traffic**: ~180° difference

We use **120°** as the cutoff. This means:
- We only flag vehicles whose heading is clearly in the opposing half (>120° away from the allowed direction)
- We tolerate up to ±60° of deviation — covering GPS jitter, gradual curves, and diagonal approaches to intersections

---

## 2. How Does the Noise Filter Work?

### Why do we need it?
GPS data is noisy. A single point can be off due to:
- Satellite signal multipath near tall buildings
- A vehicle briefly turning into a side street before correcting
- A driver doing a U-turn that's not sustained wrong-way travel

Without filtering, any of these could produce a spurious alert.

### The persistence rule
We only **confirm** a wrong-way event if **3 or more consecutive GPS points** from the same vehicle exceed the bearing threshold.

At a typical GPS sampling rate of one point every 2 seconds, 3 points = ~6 seconds of sustained wrong-way travel. That's enough to rule out momentary noise, and quick enough to catch a real incident before it causes danger.

### Implementation
1. Each point is first labeled `candidate` or `normal`
2. For each vehicle, we scan for runs of consecutive `candidate` points
3. Runs ≥ 3 → all points in the run become `wrong_way`
4. Shorter runs → reset to `normal` (noise suppressed)

---

## 3. Known False Positive Scenarios

### Scenario A: U-turns
**Situation:** A vehicle legally U-turns on a bidirectional road, briefly driving against the flow in a one-way segment adjacent to the intersection.

**Why it triggers:** For 2–3 GPS points the vehicle's bearing will be ~180° from the road direction.

**Mitigation in production:**
- Increase `PERSISTENCE_WINDOW` to 5 (10 seconds) near intersections
- Cross-reference with OSM turn restriction data — if a U-turn is permitted at this junction, suppress the alert
- Use vehicle speed: very low speed + wrong-way bearing → likely maneuvering, not a true incident

---

### Scenario B: Roadworks / Temporary Contraflow
**Situation:** Road authorities have temporarily reversed a one-way street due to construction. OSM hasn't been updated yet.

**Why it triggers:** The vehicle is following the temporary allowed direction, but our graph still says it's one-way in the original direction.

**Mitigation in production:**
- Ingest real-time roadworks data feeds (e.g. HERE Traffic, TomTom incidents)
- Allow operator override: suppress alerts on flagged road segments during known roadworks windows
- Corroborate across multiple vehicles — if 5+ vehicles all "go wrong way" on the same segment, it's probably a data issue, not 5 incidents

---

### Scenario C: GPS Drift Near Tunnels/Overpasses
**Situation:** A vehicle exits a tunnel or passes under an overpass. GPS position jumps by 20–40m, placing the point on the wrong side of a road divider or on an adjacent one-way street.

**Why it triggers:** The snapped road segment has a completely different bearing than where the vehicle actually is.

**Mitigation in production:**
- Filter out GPS points where horizontal accuracy (HDOP) is low
- Use map matching (HMM-based) instead of simple nearest-edge snapping — it considers trajectory continuity
- Add a "confidence" score based on snap distance; discard points snapped > 20m from the road centreline

---

### Scenario D: Divided Highways
**Situation:** A dual carriageway (e.g. a 6-lane road) has opposing lanes mapped as separate one-way edges in OSM. GPS imprecision places a vehicle on the wrong carriageway edge.

**Mitigation in production:**
- Filter by road type: raise the threshold or increase the persistence window for roads tagged `highway=motorway` or `highway=trunk`
- Use lane-level map data where available

---

## 4. What We Would Add in a Production System

| Enhancement | Benefit |
|-------------|---------|
| Map matching (Viterbi / HMM) | More accurate road snapping using trajectory history |
| Real-time incident feed | Suppress alerts during known roadworks |
| Speed filter | Low speed + wrong bearing = maneuvering, not incident |
| Confidence scores | Surface uncertainty instead of hard yes/no labels |
| Webhook alerts | Push a notification when a confirmed wrong-way event starts |
| Geofence exclusion zones | Mark known problem areas (tunnels, airports) for special handling |