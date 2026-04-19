"""
run_detection.py  (v3 — multi-threshold)
-----------------------------------------
Entry point. One command runs the full pipeline at THREE angle thresholds
(90°, 120°, 150°) and writes a separate labeled_traces_{threshold}.csv for
each, plus the segment_alerts.json produced by the default (120°) run.

All three output CSVs have identical columns to the original labeled_traces.csv:
    vehicle_id, lat, lon, timestamp, bearing, speed_kmh, road_bearing,
    angular_diff, snap_distance_m, edge_id, suppressed_by, label,
    run_length, confidence

Usage:
    python src/run_detection.py
    python src/run_detection.py --input data/raw/gps_traces.csv
    python src/run_detection.py --thresholds 90 120 150
"""

import os
import argparse
import pandas as pd

from road_graph import load_graph
from detector import run_detection
from segment_aggregator import export_for_personB

# ── DEFAULTS ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT      = "../data/raw/gps_traces.csv"
DEFAULT_OUTPUT_DIR = "../data/output"
DEFAULT_ALERTS_OUT        = "../data/output/segment_alerts.json"
DEFAULT_EXPLAINABILITY_OUT = "../data/output/explainability_index.json"
DEFAULT_THRESHOLDS = [90, 120, 150]

# The threshold whose labeled CSV is also treated as the canonical
# labeled_traces.csv (used by inject_data.py / the map)
CANONICAL_THRESHOLD = 120

# Exact column order matching the original labeled_traces.csv
LABELED_TRACES_COLUMNS = [
    "vehicle_id", "lat", "lon", "timestamp", "bearing", "speed_kmh",
    "road_bearing", "angular_diff", "snap_distance_m", "edge_id",
    "suppressed_by", "label", "run_length", "confidence",
]


import json


# ── EXPLAINABILITY INDEX ──────────────────────────────────────────────────────

def _score_components(row: pd.Series, threshold: int) -> dict:
    """
    Re-derive the three confidence sub-scores for one wrong_way row so they
    can be stored individually in the explainability record.
    Mirrors the formula in detector.compute_confidence() exactly.
    """
    headroom = max(180 - threshold, 1)

    score_a = round(
        max(0.0, min(50.0, (row["angular_diff"] - threshold) / headroom * 50)), 1
    )
    score_b = round(
        min(1.0, row["run_length"] / 10) * 30, 1
    )
    snap = row["snap_distance_m"]
    if pd.isna(snap) or snap is None:
        score_c = 0.0
    else:
        score_c = round(max(0.0, 1 - snap / 30) * 20, 1)

    return {"score_a_angular":      score_a,
            "score_b_persistence":  score_b,
            "score_c_snap_quality": score_c}


def export_explainability_index(
    results: dict,      # {threshold: labeled_df}  from main()
    output_path: str,
) -> None:
    """
    Merge wrong_way rows from every threshold run into one JSON file.

    Schema per record
    -----------------
    key                 : "{vehicle_id}_{timestamp}_{threshold}" — globally unique
    vehicle_id          : str
    timestamp           : str
    threshold_used      : int   — angle threshold that produced this detection
    vehicle_bearing     : float — GPS bearing of the vehicle
    road_bearing        : float — bearing of the snapped road edge
    angular_diff        : float — absolute angular difference
    run_length          : int   — consecutive wrong_way points in this run
    snap_distance_m     : float — metres from GPS point to road centreline
    edge_id             : str   — OSM edge that was snapped to
    suppressed_by       : str | null
    confidence          : float — total score (0–100)
    score_a_angular     : float — angular component    (max 50 pts)
    score_b_persistence : float — persistence component (max 30 pts)
    score_c_snap_quality: float — snap quality component (max 20 pts)
    threshold_comparison: list  — what happened to this (vehicle, timestamp)
                                  at every other threshold in the run

    Key design notes
    ----------------
    - Key includes threshold so the same GPS point detected at two different
      thresholds appears as two distinct records — they have different math
      and different confidence scores, so collapsing them would lose information.
    - threshold_comparison lets you see, for any detection, whether the other
      thresholds also caught it and at what confidence — useful for sensitivity
      analysis and driving a cross-threshold toggle in the frontend.
    - Records are sorted by (vehicle_id, timestamp, threshold) so diffs between
      pipeline runs are human-readable.
    """
    thresholds = sorted(results.keys())

    # Build lookup: (vehicle_id, timestamp) → {threshold: row}
    # so threshold_comparison fills in O(n) without nested re-scans.
    lookup: dict[tuple, dict] = {}
    for t, df in results.items():
        ww = df[df["label"] == "wrong_way"]
        for _, row in ww.iterrows():
            pair = (row["vehicle_id"], str(row["timestamp"]))
            if pair not in lookup:
                lookup[pair] = {}
            lookup[pair][t] = row

    records = []
    for t in thresholds:
        ww = results[t][results[t]["label"] == "wrong_way"]

        for _, row in ww.iterrows():
            pair  = (row["vehicle_id"], str(row["timestamp"]))
            comps = _score_components(row, t)

            # Cross-threshold comparison for this (vehicle, timestamp) pair
            comparison = []
            for other_t in thresholds:
                if other_t == t:
                    continue
                other_row = lookup.get(pair, {}).get(other_t)
                if other_row is not None:
                    other_comps = _score_components(other_row, other_t)
                    comparison.append({
                        "threshold":             other_t,
                        "detected":              True,
                        "confidence":            round(float(other_row["confidence"]), 1),
                        **other_comps,
                    })
                else:
                    comparison.append({
                        "threshold":             other_t,
                        "detected":              False,
                        "confidence":            None,
                        "score_a_angular":       None,
                        "score_b_persistence":   None,
                        "score_c_snap_quality":  None,
                    })

            snap = row["snap_distance_m"]
            records.append({
                "key":               f"{row['vehicle_id']}_{row['timestamp']}_{t}",
                "vehicle_id":        str(row["vehicle_id"]),
                "timestamp":         str(row["timestamp"]),
                "threshold_used":    t,
                "vehicle_bearing":   round(float(row["bearing"]), 1),
                "road_bearing":      round(float(row["road_bearing"]), 1),
                "angular_diff":      round(float(row["angular_diff"]), 1),
                "run_length":        int(row["run_length"]),
                "snap_distance_m":   round(float(snap), 1) if pd.notna(snap) else None,
                "edge_id":           str(row["edge_id"]),
                "suppressed_by":     row["suppressed_by"] if pd.notna(row["suppressed_by"]) else None,
                "confidence":        round(float(row["confidence"]), 1),
                **comps,
                "threshold_comparison": comparison,
            })

    records.sort(key=lambda r: (r["vehicle_id"], r["timestamp"], r["threshold_used"]))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=True)

    # Summary table
    print(f"\n── Explainability index {'─' * 28}")
    print(f"  Records total : {len(records)}")
    by_t: dict[int, list] = {}
    for r in records:
        by_t.setdefault(r["threshold_used"], []).append(r)
    for t in thresholds:
        rows_t = by_t.get(t, [])
        confs  = [r["confidence"] for r in rows_t]
        mean_c = f"{sum(confs)/len(confs):.1f}" if confs else "—"
        print(f"  threshold={t:>3}° : {len(rows_t):>4} wrong_way records  "
              f"mean_confidence={mean_c}")
    print(f"  Exported      : {output_path}")


# ── VALIDATION ────────────────────────────────────────────────────────────────
def validate_input(df: pd.DataFrame) -> None:
    required_cols = {"vehicle_id", "lat", "lon", "timestamp", "bearing"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    if "speed_kmh" not in df.columns:
        print("[WARNING] No 'speed_kmh' column — speed filter will be skipped.")

    print(f"[run_detection] {len(df)} rows, {df['vehicle_id'].nunique()} vehicles loaded")


# ── COLUMN ENFORCEMENT ────────────────────────────────────────────────────────
def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee the output DataFrame has exactly the columns in LABELED_TRACES_COLUMNS
    and in that order.  Missing columns are filled with None so that downstream
    code (inject_data.py, the map) never sees a KeyError regardless of which
    columns detector.py happens to produce at a given threshold.
    """
    for col in LABELED_TRACES_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[LABELED_TRACES_COLUMNS]


# ── PER-THRESHOLD RUN ─────────────────────────────────────────────────────────
def run_single_threshold(df: pd.DataFrame, G, threshold: int,
                         output_dir: str) -> pd.DataFrame:
    """
    Run detection at one angle_threshold, enforce schema, write CSV, return df.
    """
    print(f"\n── Threshold {threshold}° {'─' * 38}")

    labeled_df = run_detection(df, G, angle_threshold=threshold)
    labeled_df = enforce_schema(labeled_df)

    out_path = os.path.join(output_dir, f"labeled_traces_{threshold}.csv")
    labeled_df.to_csv(out_path, index=False)

    ww_count = (labeled_df["label"] == "wrong_way").sum()
    print(f"  Wrong-way points : {ww_count} / {len(labeled_df)}")
    print(f"  Exported         : {out_path}")

    return labeled_df


# ── SUMMARY PRINTER ───────────────────────────────────────────────────────────
def print_summary(labeled_df: pd.DataFrame, summary: dict, threshold: int) -> None:
    s = summary.get("summary", {})
    print(f"\n── Detection Summary (canonical threshold = {threshold}°) {'─' * 10}")
    print(f"  Total points:          {s.get('total_points', len(labeled_df))}")
    print(f"  Wrong-way points:      {s.get('wrong_way_points', (labeled_df.label=='wrong_way').sum())}")
    print(f"  Vehicles (total):      {s.get('vehicles_total', labeled_df.vehicle_id.nunique())}")
    print(f"  Vehicles (wrong-way):  {s.get('vehicles_wrong_way', '—')}")
    print(f"  Segments affected:     {s.get('segments_affected', '—')}")
    print(f"  HIGH risk segments:    {s.get('high_risk_segments', '—')}")

    print(f"\n── Per-vehicle breakdown {'─' * 27}")
    breakdown = (
        labeled_df.groupby(["vehicle_id", "label"])
        .size()
        .unstack(fill_value=0)
    )
    print(breakdown.to_string())

    ww = labeled_df[labeled_df["label"] == "wrong_way"]
    if not ww.empty:
        print(f"\n── Confidence stats on wrong_way points {'─' * 12}")
        print(ww["confidence"].describe().round(1).to_string())


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(
    input_path: str = DEFAULT_INPUT,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    alerts_out: str = DEFAULT_ALERTS_OUT,
    explainability_out: str = DEFAULT_EXPLAINABILITY_OUT,
    thresholds: list[int] = None,
) -> dict[int, pd.DataFrame]:
    """
    Returns a dict mapping each threshold → its labeled DataFrame.
    The canonical threshold's DataFrame is also written as labeled_traces.csv
    (the file inject_data.py expects).
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    print(f"\n{'=' * 50}")
    print(f"  Wrong-Way Detection Pipeline v3")
    print(f"  Thresholds : {thresholds}")
    print(f"{'=' * 50}\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load + validate
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    validate_input(df)

    # 2. Road graph (loaded once, shared across all threshold runs)
    G = load_graph()

    # 3. Run each threshold
    results: dict[int, pd.DataFrame] = {}
    for t in thresholds:
        results[t] = run_single_threshold(df, G, t, output_dir)

    # 4. Canonical run: export segment_alerts.json + canonical labeled_traces.csv
    canonical_threshold = CANONICAL_THRESHOLD if CANONICAL_THRESHOLD in thresholds else thresholds[0]
    canonical_df = results[canonical_threshold]

    summary = export_for_personB(
        canonical_df,
        alert_path=alerts_out,
        points_path=os.path.join(output_dir, "labeled_traces.csv"),
    )

    # 5. Export explainability index (all thresholds → one JSON)
    export_explainability_index(results, output_path=explainability_out)

    # 6. Print summary
    print_summary(canonical_df, summary, canonical_threshold)

    # 7. Cross-threshold comparison table
    print(f"\n── Cross-threshold comparison {'─' * 22}")
    print(f"  {'Threshold':>10}  {'WW points':>10}  {'Mean conf':>10}  {'Vehicles WW':>12}")
    for t, ldf in results.items():
        ww = ldf[ldf["label"] == "wrong_way"]
        mean_conf = f"{ww['confidence'].mean():.1f}" if not ww.empty else "—"
        ww_veh = ww["vehicle_id"].nunique() if not ww.empty else 0
        marker = " ← canonical" if t == canonical_threshold else ""
        print(f"  {t:>9}°  {len(ww):>10}  {mean_conf:>10}  {ww_veh:>12}{marker}")

    print(f"\n{'=' * 50}")
    print(f"  Outputs written to: {output_dir}/")
    for t in thresholds:
        print(f"    labeled_traces_{t}.csv")
    print(f"    labeled_traces.csv  (canonical = {canonical_threshold}°)")
    print(f"    {os.path.basename(alerts_out)}")
    print(f"    {os.path.basename(explainability_out)}")
    print(f"{'=' * 50}\n")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wrong-Way Vehicle Detector v3")
    parser.add_argument("--input",      default=DEFAULT_INPUT,
                        help="Path to input GPS traces CSV")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory for all output CSVs")
    parser.add_argument("--alerts-out",          default=DEFAULT_ALERTS_OUT,
                        help="Path for segment_alerts.json")
    parser.add_argument("--explainability-out",  default=DEFAULT_EXPLAINABILITY_OUT,
                        help="Path for explainability_index.json")
    parser.add_argument("--thresholds", nargs="+", type=int,
                        default=DEFAULT_THRESHOLDS,
                        metavar="DEG",
                        help="One or more angle thresholds in degrees (default: 90 120 150)")
    args = parser.parse_args()

    main(
        input_path=args.input,
        output_dir=args.output_dir,
        alerts_out=args.alerts_out,
        explainability_out=args.explainability_out,
        thresholds=args.thresholds,
    )