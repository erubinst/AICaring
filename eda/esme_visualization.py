"""
location_overlap.py
════════════════════
Visualises shared locations between two people's Google Maps Timeline
exports (new on-device format: semanticSegments), with week-by-week
filtering built into the output HTML map.

WHAT IT DOES
────────────
1. Parses both Timeline.json files (visits + approach routes).
2. Clusters all visits globally by geography (DBSCAN / haversine) so the
   same real-world place always maps to the same cluster ID across weeks.
3. Finds clusters visited by BOTH people at any point.
4. Builds a single interactive HTML map with a week selector at the top —
   use the ← → buttons or the dropdown to step through weeks. Only that
   week's visits and routes are shown; the shared-cluster rings stay
   visible at all times as a reference frame.

USAGE
─────
    python location_overlap.py person1.json person2.json [options]

OPTIONS
───────
    --p1-name  NAME    Label for person 1 (default: "Person 1")
    --p2-name  NAME    Label for person 2 (default: "Person 2")
    --radius-m FLOAT   Cluster radius in metres (default: 150)
    --output   FILE    Output HTML file (default: location_overlap.html)
    --csv      FILE    Also write a CSV of all shared visits (optional)

EXAMPLE
───────
    python location_overlap.py alice.json bob.json \\
        --p1-name Alice --p2-name Bob --radius-m 200 --output map.html
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import folium

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
EARTH_RADIUS_KM = 6371.0
P1_COLOR        = "#e74c3c"   # red
P2_COLOR        = "#2980b9"   # blue
CLUSTER_COLOR   = "#27ae60"   # green


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_ts(s: str) -> datetime | None:
    if not s:
        return None
    s = re.sub(r'[Zz]$', '', s.strip())
    s = re.sub(r'[+-]\d{2}:\d{2}$', '', s)
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_point(s: str) -> tuple[float, float] | None:
    if not s:
        return None
    nums = re.findall(r"[-+]?\d+\.?\d*", s)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


def fmt_time(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M") if dt else "unknown"


def iso_week_label(dt: datetime | None) -> str:
    """Return 'YYYY-Www' e.g. '2024-W11'."""
    if dt is None:
        return "unknown"
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_timeline(path: str) -> tuple[list[dict], list[dict]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("semanticSegments", [])
    if not segments:
        print(f"  ⚠  No semanticSegments found in {path}")

    visits: list[dict] = []
    routes: list[dict] = []

    for i, seg in enumerate(segments):

        if "visit" in seg:
            top       = seg["visit"].get("topCandidate", {})
            loc       = top.get("placeLocation", {})
            raw_coord = loc.get("latLng") or loc.get("point") or seg.get("startLocation")
            coords    = parse_point(raw_coord) if isinstance(raw_coord, str) else None
            if coords is None:
                continue
            lat, lon = coords
            start = parse_ts(seg.get("startTime"))
            visits.append({
                "lat":           lat,
                "lon":           lon,
                "place_id":      top.get("placeId", ""),
                "semantic_type": top.get("semanticType", ""),
                "start":         start,
                "end":           parse_ts(seg.get("endTime")),
                "week":          iso_week_label(start),
                "segment_index": i,
            })

        elif "timelinePath" in seg:
            points = []
            for pt in seg["timelinePath"]:
                raw = pt.get("point") or pt.get("latLng")
                if raw:
                    c = parse_point(raw)
                    if c:
                        points.append(c)
            if len(points) >= 2:
                routes.append({
                    "points":         points,
                    "start":          parse_ts(seg.get("startTime")),
                    "end":            parse_ts(seg.get("endTime")),
                    "activity_type":  seg.get("activity", {})
                                         .get("topCandidate", {})
                                         .get("type", ""),
                    "segment_index":  i,
                    "next_visit_idx": None,
                })

    seg_to_visit_idx = {v["segment_index"]: vi for vi, v in enumerate(visits)}
    max_seg = max(seg_to_visit_idx, default=0)
    for route in routes:
        for j in range(route["segment_index"] + 1, max_seg + 2):
            if j in seg_to_visit_idx:
                route["next_visit_idx"] = seg_to_visit_idx[j]
                break

    return visits, routes


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def cluster_visits(all_visits: list[dict], radius_m: float) -> np.ndarray:
    if not all_visits:
        return np.array([], dtype=int)
    coords_rad = np.radians([[v["lat"], v["lon"]] for v in all_visits])
    eps_rad    = (radius_m / 1000.0) / EARTH_RADIUS_KM
    labels     = DBSCAN(eps=eps_rad, min_samples=1,
                        algorithm="ball_tree", metric="haversine").fit_predict(coords_rad)
    max_lbl = int(labels.max()) if len(labels) else -1
    for idx in range(len(labels)):
        if labels[idx] == -1:
            max_lbl += 1
            labels[idx] = max_lbl
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# SHARED-CLUSTER FINDER
# ══════════════════════════════════════════════════════════════════════════════

def find_shared_clusters(
    p1_visits, p2_visits,
    p1_routes, p2_routes,
    all_labels: np.ndarray,
    n_p1: int,
) -> dict:
    p1_labels = all_labels[:n_p1]
    p2_labels = all_labels[n_p1:]

    p1_by_cluster: dict[int, list] = defaultdict(list)
    p2_by_cluster: dict[int, list] = defaultdict(list)
    for v, lbl in zip(p1_visits, p1_labels):
        p1_by_cluster[lbl].append(v)
    for v, lbl in zip(p2_visits, p2_labels):
        p2_by_cluster[lbl].append(v)

    p1_route_map = {r["next_visit_idx"]: r for r in p1_routes if r["next_visit_idx"] is not None}
    p2_route_map = {r["next_visit_idx"]: r for r in p2_routes if r["next_visit_idx"] is not None}

    shared = {}
    for cid in set(p1_by_cluster) & set(p2_by_cluster):
        p1_vs = p1_by_cluster[cid]
        p2_vs = p2_by_cluster[cid]
        shared[cid] = {
            "p1_visits":    p1_vs,
            "p2_visits":    p2_vs,
            "p1_routes":    [p1_route_map.get(p1_visits.index(v)) for v in p1_vs],
            "p2_routes":    [p2_route_map.get(p2_visits.index(v)) for v in p2_vs],
            "centre_lat":   float(np.mean([v["lat"] for v in p1_vs + p2_vs])),
            "centre_lon":   float(np.mean([v["lon"] for v in p1_vs + p2_vs])),
            "total_visits": len(p1_vs) + len(p2_vs),
        }
    return shared


# ══════════════════════════════════════════════════════════════════════════════
# MAP BUILDER  (week-filtered via embedded JS)
# ══════════════════════════════════════════════════════════════════════════════

def build_map(shared_clusters: dict, p1_name: str, p2_name: str, output_path: str):
    if not shared_clusters:
        print("  ⚠  No shared clusters found.")
        return

    # Collect all ISO weeks that appear in shared-cluster visits
    all_weeks: set[str] = set()
    for info in shared_clusters.values():
        for v in info["p1_visits"] + info["p2_visits"]:
            if v["week"] != "unknown":
                all_weeks.add(v["week"])
    sorted_weeks = sorted(all_weeks)

    # Base map
    lats = [c["centre_lat"] for c in shared_clusters.values()]
    lons = [c["centre_lon"] for c in shared_clusters.values()]
    m = folium.Map(location=[np.mean(lats), np.mean(lons)],
                   zoom_start=13, tiles="CartoDB positron")

    # Always-visible cluster rings
    for cid, info in sorted(shared_clusters.items()):
        r = max(14, min(40, 14 + info["total_visits"] * 3))
        popup_html = (
            f"<b>Shared cluster #{cid}</b><br>"
            f"{p1_name}: {len(info['p1_visits'])} total visit(s)<br>"
            f"{p2_name}: {len(info['p2_visits'])} total visit(s)"
        )
        folium.CircleMarker(
            location=[info["centre_lat"], info["centre_lon"]],
            radius=r,
            color=CLUSTER_COLOR,
            fill=True,
            fill_color=CLUSTER_COLOR,
            fill_opacity=0.15,
            weight=2.5,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"Cluster #{cid}",
        ).add_to(m)

    # Build week-keyed feature list for JS
    week_data: dict[str, list] = defaultdict(list)

    for cid, info in sorted(shared_clusters.items()):

        def add_visit(visit, person, color):
            place = visit.get("place_id") or visit.get("semantic_type") or "—"
            week_data[visit["week"]].append({
                "type":    "pin",
                "person":  person,
                "color":   color,
                "lat":     visit["lat"],
                "lon":     visit["lon"],
                "popup":   (f"<b>{person}</b> · cluster #{cid}<br>"
                            f"Place: {place}<br>"
                            f"Arrived: {fmt_time(visit['start'])}<br>"
                            f"Left: {fmt_time(visit['end'])}"),
                "tooltip": f"{person} @ cluster #{cid}",
            })

        def add_route(route, visit, person, color):
            if route is None:
                return
            place = visit.get("place_id") or visit.get("semantic_type") or "—"
            week_data[visit["week"]].append({
                "type":    "route",
                "person":  person,
                "color":   color,
                "points":  route["points"],
                "end_dot": route["points"][-1],
                "popup":   (f"<b>{person}</b> approach → cluster #{cid}<br>"
                            f"Mode: {route['activity_type'] or 'unknown'}<br>"
                            f"Departed: {fmt_time(route['start'])}<br>"
                            f"Arrived: {fmt_time(visit['start'])}<br>"
                            f"Place: {place}"),
                "tooltip": f"{person} approach to cluster #{cid}",
            })

        for v, r in zip(info["p1_visits"], info["p1_routes"]):
            add_visit(v, p1_name, P1_COLOR)
            add_route(r, v, p1_name, P1_COLOR)

        for v, r in zip(info["p2_visits"], info["p2_routes"]):
            add_visit(v, p2_name, P2_COLOR)
            add_route(r, v, p2_name, P2_COLOR)

    # Inject CSS panel
    panel_css = f"""
<style>
#week-ctrl {{
  display: none;
  position: fixed;
  top: 14px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 2000;
  background: white;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.22);
  padding: 9px 16px;
  font-family: Arial, sans-serif;
  font-size: 13px;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  justify-content: center;
  min-width: 340px;
}}
#week-ctrl button {{
  border: 1px solid #ccc;
  background: #f5f5f5;
  border-radius: 5px;
  padding: 4px 12px;
  cursor: pointer;
  font-size: 15px;
  line-height: 1;
}}
#week-ctrl button:hover:not(:disabled) {{ background: #e0e0e0; }}
#week-ctrl button:disabled {{ opacity: 0.3; cursor: default; }}
#week-select {{
  border: 1px solid #ccc;
  border-radius: 5px;
  padding: 3px 7px;
  font-size: 13px;
}}
#week-label {{
  font-weight: bold;
  font-size: 14px;
  min-width: 85px;
  text-align: center;
}}
#week-stats {{
  font-size: 11px;
  color: #555;
  width: 100%;
  text-align: center;
  margin-top: 2px;
}}
</style>
<div id="week-ctrl">
  <button id="prev-week" onclick="prevWeek()">&#8592;</button>
  <select id="week-select" onchange="showWeek(parseInt(this.value))"></select>
  <span id="week-label"></span>
  <button id="next-week" onclick="nextWeek()">&#8594;</button>
  <div id="week-stats"></div>
</div>
"""

    # Inject JS
    js_block = f"""
<script>
var WEEKS     = {json.dumps(sorted_weeks)};
var WEEK_DATA = {json.dumps(week_data)};
var P1_NAME   = {json.dumps(p1_name)};
var P2_NAME   = {json.dumps(p2_name)};
var currentWeekIdx   = 0;
var activeLayerGroup = null;

document.addEventListener("DOMContentLoaded", function() {{
  setTimeout(initWeekControls, 400);
}});

function initWeekControls() {{
  var mapObj = null;
  for (var key in window) {{
    if (key.startsWith("map_") && window[key] && window[key].addLayer) {{
      mapObj = window[key]; break;
    }}
  }}
  if (!mapObj) {{ console.warn("Leaflet map not found"); return; }}
  window._leafletMap = mapObj;
  activeLayerGroup = L.layerGroup().addTo(mapObj);

  var sel = document.getElementById("week-select");
  WEEKS.forEach(function(w, i) {{
    var opt = document.createElement("option");
    opt.value = i; opt.text = w;
    sel.appendChild(opt);
  }});

  document.getElementById("week-ctrl").style.display = "flex";
  showWeek(0);
}}

function showWeek(idx) {{
  if (!WEEKS.length) return;
  idx = Math.max(0, Math.min(idx, WEEKS.length - 1));
  currentWeekIdx = idx;
  var week     = WEEKS[idx];
  var features = WEEK_DATA[week] || [];

  document.getElementById("week-select").value = idx;
  document.getElementById("week-label").textContent = week;
  document.getElementById("prev-week").disabled = (idx === 0);
  document.getElementById("next-week").disabled = (idx === WEEKS.length - 1);

  var p1c = features.filter(function(f){{return f.type==="pin"&&f.person===P1_NAME;}}).length;
  var p2c = features.filter(function(f){{return f.type==="pin"&&f.person===P2_NAME;}}).length;
  document.getElementById("week-stats").textContent =
    P1_NAME + ": " + p1c + " visit(s)   ·   " + P2_NAME + ": " + p2c + " visit(s)";

  activeLayerGroup.clearLayers();

  features.forEach(function(f) {{
    if (f.type === "pin") {{
      var mk = L.circleMarker([f.lat, f.lon], {{
        radius: 8, color: f.color, fillColor: f.color,
        fillOpacity: 0.85, weight: 1.5
      }}).bindPopup(f.popup).bindTooltip(f.tooltip);
      activeLayerGroup.addLayer(mk);

    }} else if (f.type === "route") {{
      var ll = f.points.map(function(p){{ return [p[0], p[1]]; }});
      var ln = L.polyline(ll, {{
        color: f.color, weight: 3, opacity: 0.65, dashArray: "7 5"
      }}).bindPopup(f.popup).bindTooltip(f.tooltip);
      activeLayerGroup.addLayer(ln);
      activeLayerGroup.addLayer(
        L.circleMarker([f.end_dot[0], f.end_dot[1]], {{
          radius: 5, color: f.color, fillColor: f.color,
          fillOpacity: 0.9, weight: 1
        }})
      );
    }}
  }});
}}

function prevWeek() {{ showWeek(currentWeekIdx - 1); }}
function nextWeek() {{ showWeek(currentWeekIdx + 1); }}
</script>
"""

    legend_html = f"""
<div style="
    position:fixed; bottom:30px; left:30px; z-index:1000;
    background:white; padding:14px 18px; border-radius:10px;
    box-shadow:0 2px 8px rgba(0,0,0,0.22);
    font-family:Arial,sans-serif; font-size:13px; min-width:210px;">
  <b>📍 Location Overlap</b><br><br>
  <span style="color:{CLUSTER_COLOR};font-size:16px;">●</span> Shared cluster (all-time)<br>
  <span style="color:{P1_COLOR};font-size:16px;">●</span> {p1_name} visit / route<br>
  <span style="color:{P2_COLOR};font-size:16px;">●</span> {p2_name} visit / route<br><br>
  <b>Shared clusters: {len(shared_clusters)}</b><br>
  <b>Weeks with data: {len(sorted_weeks)}</b><br><br>
  <i style="font-size:11px;">Step through weeks using the<br>
  selector at the top of the map.<br>
  Click any marker for details.</i>
</div>
"""

    m.get_root().html.add_child(folium.Element(panel_css))
    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element(js_block))

    m.save(output_path)
    print(f"  ✅ Map saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(shared_clusters: dict, p1_name: str, p2_name: str, path: str):
    rows = []
    for cid, info in shared_clusters.items():
        for person, visits in [(p1_name, info["p1_visits"]),
                               (p2_name, info["p2_visits"])]:
            for v in visits:
                rows.append({
                    "cluster_id":    cid,
                    "person":        person,
                    "week":          v["week"],
                    "place_id":      v.get("place_id", ""),
                    "semantic_type": v.get("semantic_type", ""),
                    "arrival":       fmt_time(v["start"]),
                    "departure":     fmt_time(v["end"]),
                    "lat":           round(v["lat"], 6),
                    "lon":           round(v["lon"], 6),
                })
    pd.DataFrame(rows).sort_values(["cluster_id", "week", "person", "arrival"]).to_csv(path, index=False)
    print(f"  ✅ CSV saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Map shared locations between two Google Maps Timeline JSON files, by week."
    )
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("--p1-name",  default="Person 1")
    parser.add_argument("--p2-name",  default="Person 2")
    parser.add_argument("--radius-m", type=float, default=150,
                        help="Cluster radius in metres (default 150)")
    parser.add_argument("--output",   default="location_overlap.html")
    parser.add_argument("--csv",      default=None)
    args = parser.parse_args()

    print(f"\n📂 Parsing {args.p1_name}: {args.file1}")
    p1_visits, p1_routes = parse_timeline(args.file1)
    print(f"   → {len(p1_visits)} visits, {len(p1_routes)} routes")

    print(f"📂 Parsing {args.p2_name}: {args.file2}")
    p2_visits, p2_routes = parse_timeline(args.file2)
    print(f"   → {len(p2_visits)} visits, {len(p2_routes)} routes")

    if not p1_visits and not p2_visits:
        print("\n❌ No visits found. Confirm both files use the semanticSegments format.")
        sys.exit(1)

    n_p1       = len(p1_visits)
    all_visits = p1_visits + p2_visits
    print(f"\n🔵 Clustering {len(all_visits)} visits (radius = {args.radius_m} m)…")
    labels = cluster_visits(all_visits, radius_m=args.radius_m)
    print(f"   → {len(set(labels))} clusters formed")

    shared = find_shared_clusters(p1_visits, p2_visits, p1_routes, p2_routes, labels, n_p1)
    print(f"   → {len(shared)} cluster(s) visited by BOTH people")

    if shared:
        print("\n📋 Shared cluster summary:")
        for cid, info in sorted(shared.items()):
            wks1 = len({v["week"] for v in info["p1_visits"]})
            wks2 = len({v["week"] for v in info["p2_visits"]})
            print(f"   Cluster #{cid:3d} | "
                  f"{args.p1_name}: {len(info['p1_visits'])} visit(s) / {wks1} week(s) | "
                  f"{args.p2_name}: {len(info['p2_visits'])} visit(s) / {wks2} week(s)")

    print(f"\n🗺  Building map → {args.output}")
    build_map(shared, args.p1_name, args.p2_name, output_path=args.output)

    if args.csv:
        export_csv(shared, args.p1_name, args.p2_name, args.csv)

    print("\n✅ Done!\n")


if __name__ == "__main__":
    main()