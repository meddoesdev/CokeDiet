import pandas as pd
import json
import re
import math

def clean_df(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"WARNING: {path} not found — using empty array")
        return []
    df = df.where(pd.notnull(df), None)
    rows = df.to_dict(orient='records')
    def clean(v):
        if isinstance(v, float) and math.isnan(v): return None
        return v
    return [{k: clean(v) for k, v in r.items()} for r in rows]

# ── LOAD ALL THREE THRESHOLD DATASETS ────────────────────────
traces_90  = clean_df("data/output/labeled_traces_90.csv")
traces_120 = clean_df("data/output/labeled_traces_120.csv")
traces_150 = clean_df("data/output/labeled_traces_150.csv")

if not traces_120:
    traces_120 = clean_df("data/output/labeled_traces.csv")
    traces_90  = traces_120
    traces_150 = traces_120
    print("NOTE: Using single labeled_traces.csv for all thresholds.")

# ── LOAD SEGMENT ALERTS ───────────────────────────────────────
try:
    with open("data/output/segment_alerts.json", "r", encoding="utf-8") as f:
        alerts = json.load(f)
except FileNotFoundError:
    print("WARNING: segment_alerts.json not found")
    alerts = {"summary": {}, "segment_alerts": [], "incidents": [],
              "heatmap_points": [], "danger_zones": [], "collision_risks": []}

alerts.setdefault("danger_zones",    [])
alerts.setdefault("collision_risks", [])
summary = alerts.get("summary", {})
summary.setdefault("collision_risk_count", len(alerts["collision_risks"]))
summary.setdefault("critical_collisions",
    sum(1 for c in alerts["collision_risks"] if c.get("severity") == "CRITICAL"))
summary.setdefault("danger_zones_count", len(alerts["danger_zones"]))
alerts["summary"] = summary

# ── LOAD EXPLAINABILITY INDEX ─────────────────────────────────
try:
    with open("data/output/explainability_index.json", "r", encoding="utf-8") as f:
        explain = json.load(f)
    print(f"Explainability index: {len(explain)} records")
except FileNotFoundError:
    explain = []
    print("WARNING: explainability_index.json not found — popups will show partial data")

# ── SERIALISE ─────────────────────────────────────────────────
js_90      = json.dumps(traces_90,  indent=2, ensure_ascii=True)
js_120     = json.dumps(traces_120, indent=2, ensure_ascii=True)
js_150     = json.dumps(traces_150, indent=2, ensure_ascii=True)
js_alerts  = json.dumps(alerts,     indent=2, ensure_ascii=True)
js_explain = json.dumps(explain,    indent=2, ensure_ascii=True)

# ── READ MAP.HTML ─────────────────────────────────────────────
with open("map.html", "r", encoding="utf-8") as f:
    html = f.read()

# ── REPLACE DATA BLOCKS ───────────────────────────────────────
# Pattern matches: const NAME = [anything including empty];
# Uses \[[\s\S]*?\] to handle empty [], single-line, and multiline arrays
def replace_array(name, js_data, html):
    pattern = rf'(const {name} = )\[[\s\S]*?\];'
    replacement = f'const {name} = {js_data};'
    result, count = re.subn(pattern, replacement, html)
    if count == 0:
        print(f"WARNING: Could not find 'const {name}' in map.html")
    return result

def replace_object(name, js_data, html):
    pattern = rf'(const {name} = )\{{[\s\S]*?\}};'
    replacement = f'const {name} = {js_data};'
    result, count = re.subn(pattern, replacement, html)
    if count == 0:
        print(f"WARNING: Could not find 'const {name}' in map.html")
    return result

html = replace_array('labeledTraces90',  js_90,      html)
html = replace_array('labeledTraces120', js_120,     html)
html = replace_array('labeledTraces150', js_150,     html)
html = replace_array('explainIndex',     js_explain, html)
html = replace_object('segmentAlerts',   js_alerts,  html)

# ── WRITE BACK ────────────────────────────────────────────────
with open("map.html", "w", encoding="utf-8") as f:
    f.write(html)

# ── VALIDATION ────────────────────────────────────────────────
content = open("map.html").read()
if 'NaN' in content:
    print("WARNING: NaN values found in map.html — check CSVs for missing data")
else:
    print("Clean — no NaN in output")

print(f"\nDone. Injected:")
print(f"  90°  dataset : {len(traces_90)} rows")
print(f"  120° dataset : {len(traces_120)} rows")
print(f"  150° dataset : {len(traces_150)} rows")
print(f"  segment_alerts : {len(alerts.get('segment_alerts',[]))} segments, "
      f"{len(alerts.get('incidents',[]))} incidents")
print(f"  danger_zones   : {len(alerts.get('danger_zones',[]))}")
print(f"  collision_risks: {len(alerts.get('collision_risks',[]))}")
print(f"  explainability : {len(explain)} records")