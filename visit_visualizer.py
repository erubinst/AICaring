"""
Google Maps Visit Clustering & Heatmap Visualizer
--------------------------------------------------
Usage:
    python visualize_visits.py --data_dir ./data --output map.html

Expects one JSON file per person in --data_dir.
Each file should be a Google Takeout Semantic Location History JSON,
e.g. Takeout/Location History/Semantic Location History/2024/2024_JANUARY.json
OR a merged file containing a top-level "timelineObjects" array.

The script will:
  1. Extract all "visit" placeVisit entries from each file
  2. Cluster nearby visits (~200m radius) using DBSCAN
  3. Filter clusters visited by > 2 unique people
  4. Render an interactive Folium map with color/size scaled by visitor count
"""

import os
import json
import argparse
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import folium
from folium.plugins import MarkerCluster
from sklearn.cluster import DBSCAN

try:
    import requests
except ImportError:
    requests = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLUSTER_RADIUS_M = 200         # metres — "same block"
MIN_UNIQUE_VISITORS = 2        # show clusters with > this many visitors (i.e. 3+)
EARTH_RADIUS_M = 6_371_000     # for converting metres → radians

# Colour palette: light → dark as visitor count grows (9-stop ColorBrewer YlOrRd)
COLOUR_STOPS = [
    "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
    "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def metres_to_radians(metres: float) -> float:
    return metres / EARTH_RADIUS_M


def parse_latlng_string(s: str) -> tuple[float, float] | None:
    try:
        s = s.strip()
        if s.lower().startswith("geo:"):
            s = s[4:]
        cleaned = s.replace("°", "").replace(" ", "")
        parts = cleaned.split(",")
        if len(parts) != 2:
            return None
        return float(parts[0]), float(parts[1])
    except Exception:
        return None


def load_visits_from_file(filepath: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    visits = []

    # ── Format A: semanticSegments (2024+ wrapped format) ────────────────────
    if isinstance(data, dict) and "semanticSegments" in data:
        for seg in data["semanticSegments"]:
            visit = seg.get("visit")
            if not visit:
                continue
            top = visit.get("topCandidate", {})
            place_loc = top.get("placeLocation", {})
            latlng_str = place_loc.get("latLng", "")
            parsed = parse_latlng_string(latlng_str) if latlng_str else None
            if parsed is None:
                continue
            lat, lng = parsed
            place_id = top.get("placeId", "")
            semantic_type = top.get("semanticType", "UNKNOWN")
            name = semantic_type if semantic_type not in ("UNKNOWN", "Unknown") else (place_id or "Visit")
            visits.append({"lat": lat, "lng": lng, "name": name, "address": place_id})
        return visits

    # ── Format D: bare array with geo: URI coords ─────────────────────────────
    if isinstance(data, list) and data and "visit" in data[0] or (
        isinstance(data, list) and any("visit" in item for item in data[:10])
    ):
        for item in data:
            visit = item.get("visit")
            if not visit:
                continue
            top = visit.get("topCandidate", {})
            place_loc_raw = top.get("placeLocation", "")
            if isinstance(place_loc_raw, str):
                parsed = parse_latlng_string(place_loc_raw)
            elif isinstance(place_loc_raw, dict):
                parsed = parse_latlng_string(place_loc_raw.get("latLng", ""))
            else:
                parsed = None
            if parsed is None:
                continue
            lat, lng = parsed
            place_id = top.get("placeID") or top.get("placeId", "")
            semantic_type = top.get("semanticType", "Unknown")
            name = semantic_type if semantic_type not in ("UNKNOWN", "Unknown") else (place_id or "Visit")
            visits.append({"lat": lat, "lng": lng, "name": name, "address": place_id})
        return visits

    # ── Format B / C: timelineObjects ────────────────────────────────────────
    if isinstance(data, list):
        timeline_objects = data
    elif isinstance(data, dict) and "timelineObjects" in data:
        timeline_objects = data["timelineObjects"]
    else:
        timeline_objects = []
        for v in data.values():
            if isinstance(v, list):
                timeline_objects = v
                break

    for obj in timeline_objects:
        pv = obj.get("placeVisit") or obj.get("visit")
        if not pv:
            continue
        location = pv.get("location", {})
        lat = location.get("latitudeE7") or location.get("latitude")
        lng = location.get("longitudeE7") or location.get("longitude")
        if lat is None or lng is None:
            continue
        if isinstance(lat, int) and abs(lat) > 180:
            lat /= 1e7
        if isinstance(lng, int) and abs(lng) > 180:
            lng /= 1e7
        visits.append({
            "lat": float(lat), "lng": float(lng),
            "name": location.get("name", "Unknown Place"),
            "address": location.get("address", ""),
        })

    return visits


def load_all_visits(data_dir: str) -> dict[str, list[dict]]:
    data_dir = Path(data_dir)
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
    all_visits = {}
    for jf in json_files:
        person_id = jf.stem
        visits = load_visits_from_file(str(jf))
        print(f"  {person_id}: {len(visits)} visits loaded")
        all_visits[person_id] = visits
    return all_visits


def cluster_visits(all_visits: dict[str, list[dict]], radius_m: float = CLUSTER_RADIUS_M):
    coords = []
    person_ids = []
    place_names_flat = []
    place_ids_flat = []

    for person_id, visits in all_visits.items():
        for v in visits:
            coords.append([v["lat"], v["lng"]])
            person_ids.append(person_id)
            place_names_flat.append(v["name"])
            place_ids_flat.append(v.get("address", ""))

    if not coords:
        raise ValueError("No visit coordinates found across all files.")

    coords_np = np.radians(np.array(coords))
    eps_rad = metres_to_radians(radius_m)
    db = DBSCAN(eps=eps_rad, min_samples=1, algorithm="ball_tree", metric="haversine")
    labels = db.fit_predict(coords_np)

    cluster_data: dict[int, dict] = defaultdict(lambda: {
        "lats": [], "lngs": [], "visitors": set(), "visit_count": 0,
        "names": [], "place_ids": []
    })

    for idx, label in enumerate(labels):
        if label == -1:
            continue
        cd = cluster_data[label]
        cd["lats"].append(coords[idx][0])
        cd["lngs"].append(coords[idx][1])
        cd["visitors"].add(person_ids[idx])
        cd["visit_count"] += 1
        cd["names"].append(place_names_flat[idx])
        cd["place_ids"].append(place_ids_flat[idx])

    clusters = []
    for cid, cd in cluster_data.items():
        clusters.append({
            "cluster_id": cid,
            "lat": float(np.mean(cd["lats"])),
            "lng": float(np.mean(cd["lngs"])),
            "visitors": cd["visitors"],
            "visitor_count": len(cd["visitors"]),
            "visit_count": cd["visit_count"],
            "place_names": list(set(cd["names"])),
            "place_ids": cd["place_ids"],
        })

    return clusters


def get_colour(visitor_count: int, max_visitors: int) -> str:
    """Colour encodes UNIQUE VISITOR count — yellow → dark red."""
    if max_visitors <= 1:
        return COLOUR_STOPS[0]
    ratio = (visitor_count - 1) / (max_visitors - 1)
    idx = min(int(ratio * (len(COLOUR_STOPS) - 1)), len(COLOUR_STOPS) - 1)
    return COLOUR_STOPS[idx]


def get_radius(visit_count: int, max_visits: int) -> float:
    """Radius encodes TOTAL VISIT count — scales 6px → 28px."""
    if max_visits <= 1:
        return 6
    ratio = (visit_count - 1) / (max_visits - 1)
    return 6 + ratio * 22


# ---------------------------------------------------------------------------
# Place name cache + Google Places API lookup
# ---------------------------------------------------------------------------

CACHE_FILE = "place_name_cache.json"


def load_cache(cache_path: str) -> dict:
    p = Path(cache_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            cache = {k: v for k, v in raw.items() if not v.get("failed", False)}
            skipped = len(raw) - len(cache)
            msg = f"  Loaded {len(cache)} cached place lookups from {cache_path}"
            if skipped:
                msg += f" ({skipped} failed entries will be retried)"
            print(msg)
            return cache
        except Exception:
            pass
    return {}


def save_cache(cache: dict, cache_path: str) -> None:
    with open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, ensure_ascii=False)


def lookup_place(place_id: str, api_key: str, cache: dict) -> dict:
    if place_id in cache:
        return cache[place_id]

    if not requests:
        result = {"name": place_id, "address": "", "failed": True}
        cache[place_id] = result
        return result

    try:
        url = "https://places.googleapis.com/v1/places/" + place_id
        headers = {
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "displayName,formattedAddress",
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            status = data["error"].get("status", "UNKNOWN")
            print(f"    ℹ️  {place_id[:24]}… → {status} (will try reverse geocode)")
            result = {"name": place_id, "address": "", "failed": True}
        else:
            result = {
                "name": data.get("displayName", {}).get("text", place_id),
                "address": data.get("formattedAddress", ""),
                "failed": False,
            }
    except Exception as e:
        print(f"    ⚠️  {place_id[:24]}… → error: {e}")
        result = {"name": place_id, "address": "", "failed": True}

    cache[place_id] = result
    return result


def reverse_geocode(lat: float, lng: float, api_key: str) -> dict:
    try:
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"latlng": f"{lat},{lng}", "key": api_key}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "OK" and data.get("results"):
            best = data["results"][0]
            for component in best.get("address_components", []):
                types = component.get("types", [])
                if any(t in types for t in ("establishment", "point_of_interest", "premise")):
                    return {"name": component["long_name"], "address": best.get("formatted_address", "")}
            formatted = best.get("formatted_address", "")
            short = formatted.split(",")[0] if formatted else "Unknown location"
            return {"name": short, "address": formatted}
    except Exception as e:
        print(f"    ⚠️  Reverse geocode failed for {lat:.4f},{lng:.4f}: {e}")
    return {"name": f"{lat:.4f}, {lng:.4f}", "address": ""}


def enrich_clusters_with_names(
    clusters: list[dict], api_key: str, cache_path: str = CACHE_FILE
) -> None:
    """
    For each cluster, resolve the top 3 most frequent place IDs to real names.
    Stores only successfully resolved names in c['resolved_names'] (list of 0–3 strings).
    Place IDs that fail lookup are silently dropped — never shown as raw IDs.
    Only calls the API for IDs not already in the cache.
    """
    cache = load_cache(cache_path)
    new_lookups = 0
    save_every = 10

    # Pre-compute top-3 place IDs per cluster and collect all IDs we'll need
    cluster_top_ids = []
    all_ids_needed = set()
    for c in clusters:
        place_ids = c.get("place_ids", [])
        # Rank unique IDs by frequency, take top 3 non-empty ones
        unique_ids = sorted(set(pid for pid in place_ids if pid),
                            key=lambda pid: -place_ids.count(pid))
        top3 = unique_ids[:3]
        cluster_top_ids.append(top3)
        all_ids_needed.update(top3)

    uncached = [pid for pid in all_ids_needed if pid not in cache]
    if uncached:
        print(f"  {len(uncached)} new place IDs to resolve, "
              f"{len(all_ids_needed) - len(uncached)} already cached.")
    else:
        print(f"  All {len(all_ids_needed)} place IDs already cached — no API calls needed.")

    for c, top3 in zip(clusters, cluster_top_ids):
        resolved_names = []

        for place_id in top3:
            is_cache_hit = place_id in cache
            info = lookup_place(place_id, api_key, cache)

            if not is_cache_hit:
                new_lookups += 1
                if new_lookups % save_every == 0:
                    save_cache(cache, cache_path)
                    print(f"    … saved cache ({new_lookups} new lookups so far)")
                time.sleep(0.02)

            # Only keep if lookup succeeded and name differs from the raw place ID
            name = info.get("name", "")
            if not info.get("failed") and name and name != place_id:
                resolved_names.append(name)

        c["resolved_names"] = resolved_names  # 0–3 real resolved names

    if new_lookups > 0:
        save_cache(cache, cache_path)
        print(f"  ✅ {new_lookups} new lookups saved to {cache_path}")
    else:
        print(f"  ✅ Served entirely from cache — 0 API calls made.")


def build_map(clusters: list[dict], min_visitors: int = MIN_UNIQUE_VISITORS) -> folium.Map:
    """
    SIZE   = unique visitor count  (linear 6–28px)
    COLOUR = total visit count     (log scale, 9-stop YlOrRd)
    """
    visible = [c for c in clusters if c["visitor_count"] > min_visitors]
    print(f"\n  Total clusters: {len(clusters)}")
    print(f"  Clusters with >{min_visitors} unique visitors: {len(visible)}")
    if not visible:
        print("  Warning: no clusters meet threshold, showing all")
        visible = clusters

    avg_lat      = float(np.mean([c["lat"] for c in visible]))
    avg_lng      = float(np.mean([c["lng"] for c in visible]))
    max_visitors = max(c["visitor_count"] for c in visible)
    min_shown    = min_visitors + 1
    max_visits   = max(c["visit_count"]   for c in visible)

    TEAL = [
        "#ffffcc", "#ffeda0", "#fed976", "#feb24c",
        "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"
    ]

    def get_colour(visit_count):
        if max_visits <= 1:
            return TEAL[0]
        ratio = math.log1p(visit_count) / math.log1p(max_visits)
        idx = min(int(ratio * (len(TEAL) - 1)), len(TEAL) - 1)
        return TEAL[idx]

    def get_radius(visitor_count):
        if max_visitors <= min_shown:
            return 6.0
        ratio = (visitor_count - min_shown) / max(max_visitors - min_shown, 1)
        return 6.0 + ratio * 22.0

    # Pre-compute CSS gradient string in Python (avoids any JS string-building bugs)
    n = len(TEAL)
    css_stops = ", ".join(f"{c} {i/(n-1)*100:.1f}%" for i, c in enumerate(TEAL))
    css_gradient = f"linear-gradient(to right, {css_stops})"

    # Log-spaced tick labels for the colour legend bar
    n_ticks = 5
    tick_vals = [
        max(1, int(math.expm1(i / (n_ticks - 1) * math.log1p(max_visits))))
        for i in range(n_ticks)
    ]
    tick_labels = [f"{v:,}" for v in tick_vals]

    # Size legend: small / medium / large
    size_stops = sorted({min_shown,
                         min_shown + (max_visitors - min_shown) // 2,
                         max_visitors})

    # Serialise clusters for Leaflet JS
    cluster_records = []
    for c in sorted(visible, key=lambda x: x["visitor_count"]):
        resolved_names = c.get("resolved_names", [])  # 0–3 real names, never raw IDs
        visitors_str   = ", ".join(sorted(c["visitors"]))

        # Tooltip: first resolved name, or coords fallback
        tooltip_name = resolved_names[0] if resolved_names else f"{c['lat']:.4f}, {c['lng']:.4f}"

        # Popup title: list resolved names (bold first, smaller subsequent), or coords only
        if resolved_names:
            names_html = "".join(
                f'<div style="font-size:{"14" if i == 0 else "12"}px;'
                f'font-weight:{"700" if i == 0 else "400"};'
                f'color:{"#111" if i == 0 else "#666"};'
                f'margin-bottom:{"4" if i == 0 else "2"}px;">{name}</div>'
                for i, name in enumerate(resolved_names)
            )
            name_html = f'<div style="margin-bottom:6px;">{names_html}</div>'
        else:
            name_html = (
                f'<div style="font-size:12px;color:#999;margin-bottom:6px;">'
                f'&#128205; {c["lat"]:.5f}, {c["lng"]:.5f}</div>'
            )

        popup_inner = (
            name_html
            + '<hr style="border:none;border-top:1px solid #eee;margin:6px 0;">'
            + '<div style="font-size:12px;">'
            + f'&#128101; <b>{c["visitor_count"]}</b> unique visitors<br>'
            + f'&#128204; <b>{c["visit_count"]}</b> total visits<br>'
            + f'&#128205; {c["lat"]:.5f}, {c["lng"]:.5f}'
            + '</div>'
            + f'<div style="font-size:11px;color:#aaa;margin-top:6px;">{visitors_str}</div>'
        )
        cluster_records.append({
            "lat":      c["lat"],
            "lng":      c["lng"],
            "visitors": c["visitor_count"],
            "visits":   c["visit_count"],
            "colour":   get_colour(c["visit_count"]),
            "radius":   round(get_radius(c["visitor_count"]), 1),
            "tip":      (tooltip_name
                         + f' \u2014 {c["visitor_count"]} visitor'
                         + ('s' if c["visitor_count"] != 1 else '')
                         + f' / {c["visit_count"]:,} visit'
                         + ('s' if c["visit_count"] != 1 else '')),
            "popup":    popup_inner,
        })

    cluster_json = json.dumps(cluster_records)

    # Build size legend rows as plain HTML strings (no f-string inside JS)
    size_rows = ""
    for v in size_stops:
        px = int(get_radius(v) * 2)
        size_rows += (
            f'<div style="display:flex;align-items:center;margin:5px 0;">'
            f'<div style="width:{px}px;height:{px}px;border-radius:50%;'
            f'background:#fd8d3c;flex-shrink:0;margin-right:10px;"></div>'
            f'<span style="font-size:12px;">{v} people</span></div>'
        )

    tick_span_html = "".join(f'<span>{t}</span>' for t in tick_labels)

    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=13,
                   tiles="CartoDB dark_matter", prefer_canvas=True)

    # ── Legend panel (pure HTML, no JS needed) ────────────────────────────────
    legend_html = (
        '<div style="position:fixed;bottom:40px;left:40px;z-index:1000;'
        'background:rgba(15,15,20,0.88);border:1px solid #444;border-radius:10px;'
        'padding:14px 18px;min-width:210px;font-family:sans-serif;color:#f0f0f0;'
        'box-shadow:0 4px 20px rgba(0,0,0,.5);backdrop-filter:blur(6px);">'

        '<div style="font-size:11px;font-weight:700;letter-spacing:1px;'
        'text-transform:uppercase;margin-bottom:8px;color:#ccc;">'
        '&#127912; Colour &nbsp;&middot;&nbsp; Total Visits</div>'
        f'<div style="width:182px;height:13px;border-radius:4px;margin-bottom:4px;'
        f'background:{css_gradient};"></div>'
        f'<div style="display:flex;justify-content:space-between;width:182px;'
        f'font-size:11px;color:#aaa;margin-bottom:14px;">{tick_span_html}</div>'
        '<div style="font-size:10px;color:#555;text-align:right;'
        'width:182px;margin-top:-10px;margin-bottom:12px;">log scale</div>'

        '<div style="font-size:11px;font-weight:700;letter-spacing:1px;'
        'text-transform:uppercase;margin-bottom:6px;color:#ccc;">'
        '&#9711; Size &nbsp;&middot;&nbsp; Unique Visitors</div>'
        + size_rows
        + '</div>'
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Title ─────────────────────────────────────────────────────────────────
    title_html = (
        '<div style="position:fixed;top:20px;left:50%;transform:translateX(-50%);'
        'z-index:1000;background:rgba(15,15,20,0.88);border:1px solid #555;'
        'border-radius:8px;padding:10px 24px;font-family:sans-serif;'
        'color:#fff;font-size:15px;font-weight:600;letter-spacing:.5px;'
        'box-shadow:0 4px 20px rgba(0,0,0,.5);backdrop-filter:blur(6px);">'
        '&#128205; Shared Places &nbsp;&middot;&nbsp; 14-Person, 12-Week Study</div>'
    )
    m.get_root().html.add_child(folium.Element(title_html))

    # ── Markers via Leaflet JS ────────────────────────────────────────────────
    js = (
        '<script>\n'
        '(function(){\n'
        f'  var DATA={cluster_json};\n'
        '  function waitForMap(cb){\n'
        '    var t=setInterval(function(){\n'
        '      var el=document.querySelector(".folium-map");\n'
        '      if(!el||!window[el.id])return;\n'
        '      clearInterval(t); cb(window[el.id]);\n'
        '    },80);\n'
        '  }\n'
        '  waitForMap(function(map){\n'
        '    DATA.forEach(function(c){\n'
        '      L.circleMarker([c.lat,c.lng],{\n'
        '        radius:c.radius,color:c.colour,fillColor:c.colour,\n'
        '        fillOpacity:0.78,weight:1.5\n'
        '      })\n'
        '      .bindPopup(\'<div style="font-family:sans-serif;min-width:220px;">\'+c.popup+\'</div>\',{maxWidth:300})\n'
        '      .bindTooltip(c.tip)\n'
        '      .addTo(map);\n'
        '    });\n'
        '  });\n'
        '})();\n'
        '</script>\n'
    )
    m.get_root().html.add_child(folium.Element(js))

    return m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise shared Google Maps visits.")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output", default="shared_visits_map.html")
    parser.add_argument("--min_visitors", type=int, default=MIN_UNIQUE_VISITORS,
        help=f"Show clusters visited by more than N people (default: {MIN_UNIQUE_VISITORS})")
    parser.add_argument("--radius_m", type=float, default=CLUSTER_RADIUS_M,
        help="Clustering radius in metres. Try 50=building, 200=block, 500=neighbourhood")
    parser.add_argument("--api_key", default=None,
        help="Google Places API (New) key for resolving place names.")
    parser.add_argument("--cache_file", default=CACHE_FILE)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print("  Google Maps Visit Visualizer")
    print(f"{'='*55}")
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Cluster eps : {args.radius_m}m")
    print(f"  Min visitors: >{args.min_visitors}")
    print(f"  Output      : {args.output}")
    if args.api_key:
        print(f"  Places API  : enabled (cache: {args.cache_file})")
    else:
        print(f"  Places API  : disabled (pass --api_key to resolve names)")
    print(f"{'='*55}\n")

    print("Loading visits...")
    all_visits = load_all_visits(args.data_dir)
    total = sum(len(v) for v in all_visits.values())
    print(f"  → {total} visits across {len(all_visits)} people\n")

    print("Clustering...")
    clusters = cluster_visits(all_visits, radius_m=args.radius_m)

    if args.api_key:
        print("\nResolving place names...")
        visible_clusters = [c for c in clusters if c["visitor_count"] > args.min_visitors]
        enrich_clusters_with_names(visible_clusters, args.api_key, args.cache_file)
        resolved = {c["cluster_id"]: c for c in visible_clusters}
        for c in clusters:
            if c["cluster_id"] in resolved:
                c["resolved_names"] = resolved[c["cluster_id"]]["resolved_names"]

    print("\nBuilding map...")
    m = build_map(clusters, min_visitors=args.min_visitors)

    m.save(args.output)
    print(f"\n✅  Map saved to: {args.output}")
    print("   Open in any browser to explore.\n")


if __name__ == "__main__":
    main()