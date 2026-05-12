"""
shared_drives.py
Two-tier shared drive detection with semantic place linking.
"""

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import folium


EARTH_RADIUS_KM = 6371.0
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"
GOOGLE_PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"


_location_cache = {}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_ts(s):
    if not s:
        return None
    s = re.sub(r'[Zz]$', '', s.strip())
    s = re.sub(r'[+-]\d{2}:\d{2}$', '', s)
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_point(s):
    if not s:
        return None
    nums = re.findall(r"[-+]?\d+\.?\d*", s)
    return (float(nums[0]), float(nums[1])) if len(nums) >= 2 else None


def fmt(dt):
    return dt.strftime("%Y-%m-%d %H:%M") if dt else ""


def reverse_geocode_latlon(lat, lon):
    key = (round(lat, 6), round(lon, 6))
    cache_key = ("latlon", key)
    if cache_key in _location_cache:
        return _location_cache[cache_key]

    print(f"Reverse geocoding {lat}, {lon}...")
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={
                "lat": lat,
                "lon": lon,
                "format": "json",
                "addressdetails": 1,
                "zoom": 18,
            },
            headers={"User-Agent": "aicaring-shared-drives/1.0"},
            timeout=8,
        )
        resp.raise_for_status()
        payload = resp.json()
        name = payload.get("name") or payload.get("display_name")
        print(f" → {name}")
        _location_cache[cache_key] = name or ""
        return _location_cache[cache_key]
    except Exception:
        _location_cache[cache_key] = ""
        return ""


def geocode_place_id(place_id):
    cache_key = ("place_id", place_id)
    if cache_key in _location_cache:
        return _location_cache[cache_key]

    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key or not place_id:
        _location_cache[cache_key] = ""
        return ""

    try:
        resp = requests.get(
            GOOGLE_PLACE_DETAILS_URL,
            params={
                "place_id": place_id,
                "fields": "name,formatted_address",
                "key": api_key,
            },
            timeout=8,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") == "OK":
            result = payload.get("result", {})
            name = result.get("name") or result.get("formatted_address") or ""
            _location_cache[cache_key] = name
            return name
    except Exception:
        pass

    _location_cache[cache_key] = ""
    return ""


def resolve_location_name(visit):
    if not visit:
        return ""

    lat = visit.get("lat")
    lon = visit.get("lon")
    if lat is not None and lon is not None:
        label = reverse_geocode_latlon(lat, lon)
        if label:
            return label

    place_id = visit.get("place_id", "")
    if place_id:
        label = geocode_place_id(place_id)
        if label:
            return label

    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Timeline Parsing with Place Linking
# ──────────────────────────────────────────────────────────────────────────────

def parse_timeline(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("semanticSegments", [])
    routes = []

    # Second pass: collect routes
    for seg in segments:
        if "timelinePath" not in seg:
            continue

        pts = []
        for pt in seg["timelinePath"]:
            raw = pt.get("point") or pt.get("latLng")
            c = parse_point(raw) if raw else None
            if c:
                pts.append(c)

        if len(pts) < 2:
            continue

        start = parse_ts(seg.get("startTime"))
        end = parse_ts(seg.get("endTime"))

        routes.append({
            "points": pts,
            "origin": pts[0],
            "dest_pt": pts[-1],
            "start": start,
            "end": end,
        })

    return routes


# ──────────────────────────────────────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────────────────────────────────────

def cluster_labels(pts, radius_m):
    if not pts:
        return np.array([], dtype=int)

    coords_rad = np.radians(pts)
    eps = (radius_m / 1000.0) / EARTH_RADIUS_KM

    labels = DBSCAN(
        eps=eps,
        min_samples=1,
        algorithm="ball_tree",
        metric="haversine"
    ).fit_predict(coords_rad)

    return labels


# ──────────────────────────────────────────────────────────────────────────────
# Spatial + Temporal Metrics
# ──────────────────────────────────────────────────────────────────────────────

def median_path_dist_m(a, b):
    if not a or not b:
        return float("inf")

    lat0 = np.mean([p[0] for p in a + b])

    def xy(pts):
        x = np.array([p[1] for p in pts]) * math.cos(math.radians(lat0)) * 111320
        y = np.array([p[0] for p in pts]) * 110540
        return np.column_stack([x, y])

    xa, xb = xy(a), xy(b)
    shorter, longer = (xa, xb) if len(xa) <= len(xb) else (xb, xa)
    dists, _ = cKDTree(longer).query(shorter)

    return float(np.median(dists))


def overlap_ratio(r1, r2):
    overlap = max(
        0.0,
        (min(r1["end"], r2["end"]) -
         max(r1["start"], r2["start"])).total_seconds() / 60
    )

    dur1 = (r1["end"] - r1["start"]).total_seconds() / 60
    dur2 = (r2["end"] - r2["start"]).total_seconds() / 60

    shorter = min(dur1, dur2)
    if shorter <= 0:
        return 0.0

    return overlap / shorter


def direction_similarity(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0

    v1 = np.array(a[-1]) - np.array(a[0])
    v2 = np.array(b[-1]) - np.array(b[0])

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))


# ──────────────────────────────────────────────────────────────────────────────
# Confidence + Classification
# ──────────────────────────────────────────────────────────────────────────────

def compute_confidence(overlap_ratio, med_dist, path_threshold, dir_sim):
    overlap_score = min(overlap_ratio, 1.0)
    path_score = max(0.0, 1 - (med_dist / path_threshold))
    direction_score = (dir_sim + 1) / 2  # normalize -1→1 to 0→1

    return round(
        0.45 * overlap_score +
        0.35 * path_score +
        0.20 * direction_score,
        2
    )


def classify_trip(confidence):
    if confidence >= 0.63:
        return "together"
    elif confidence >= 0.55:
        return "parallel"
    else:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Matching Engine
# ──────────────────────────────────────────────────────────────────────────────

def find_shared_drives(
    p1_routes,
    p2_routes,
    radius_m,
    path_dist_m,
    min_overlap_ratio=0.25,
):
    p1_valid = [r for r in p1_routes if r["start"] and r["end"]]
    p2_valid = [r for r in p2_routes if r["start"] and r["end"]]

    if not p1_valid or not p2_valid:
        return []

    all_orig = [r["origin"] for r in p1_valid] + \
               [r["origin"] for r in p2_valid]

    all_dest = [r["dest_pt"] for r in p1_valid] + \
               [r["dest_pt"] for r in p2_valid]

    orig_lbl = cluster_labels(all_orig, radius_m)
    dest_lbl = cluster_labels(all_dest, radius_m)

    n1 = len(p1_valid)
    matches = []

    for i1, r1 in enumerate(p1_valid):
        for i2, r2 in enumerate(p2_valid):

            if orig_lbl[i1] != orig_lbl[n1 + i2]:
                continue

            if dest_lbl[i1] != dest_lbl[n1 + i2]:
                continue

            o_ratio = overlap_ratio(r1, r2)
            if o_ratio < min_overlap_ratio:
                continue

            med = median_path_dist_m(r1["points"], r2["points"])
            dir_sim = direction_similarity(r1["points"], r2["points"])

            conf = compute_confidence(o_ratio, med, path_dist_m, dir_sim)
            category = classify_trip(conf)

            if category is None:
                continue

            matches.append({
                "r1": r1,
                "r2": r2,
                "date": r1["start"].strftime("%Y-%m-%d"),
                "overlap_ratio": round(o_ratio, 2),
                "med_path_dist_m": round(med, 1),
                "direction_similarity": round(dir_sim, 2),
                "confidence": conf,
                "category": category,
            })

    return sorted(matches, key=lambda x: x["r1"]["start"])


# ──────────────────────────────────────────────────────────────────────────────
# CSV Output
# ──────────────────────────────────────────────────────────────────────────────

def build_csv(matches, p1_name, p2_name, path):
    rows = []

    for i, m in enumerate(matches):
        r1, r2 = m["r1"], m["r2"]

        p1_origin_name = reverse_geocode_latlon(r1["origin"][0], r1["origin"][1])
        p1_dest_name = reverse_geocode_latlon(r1["dest_pt"][0], r1["dest_pt"][1])

        p2_origin_name = reverse_geocode_latlon(r2["origin"][0], r2["origin"][1])
        p2_dest_name = reverse_geocode_latlon(r2["dest_pt"][0], r2["dest_pt"][1])

        rows.append({
            "match_id": i + 1,
            "date": m["date"],

            f"{p1_name}_depart": fmt(r1["start"]),
            f"{p1_name}_arrive": fmt(r1["end"]),
            f"{p1_name}_origin_location_name": p1_origin_name,
            f"{p1_name}_dest_location_name": p1_dest_name,

            f"{p2_name}_depart": fmt(r2["start"]),
            f"{p2_name}_arrive": fmt(r2["end"]),
            f"{p2_name}_origin_location_name": p2_origin_name,
            f"{p2_name}_dest_location_name": p2_dest_name,

            "confidence": m["confidence"],
            # "category": m["category"],
        })

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"✅ CSV → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Map Output
# ──────────────────────────────────────────────────────────────────────────────

def build_map(matches, p1_name, p2_name, path):
    if not matches:
        print("No matches found.")
        return

    all_pts = []
    for m in matches:
        all_pts += m["r1"]["points"] + m["r2"]["points"]

    centre = [
        float(np.mean([p[0] for p in all_pts])),
        float(np.mean([p[1] for p in all_pts]))
    ]

    m_map = folium.Map(location=centre, zoom_start=12)

    for m in matches:

        if m["category"] == "together":
            color = "blue"
            opacity = 1.0
        else:
            color = "red"
            opacity = 0.6

        tooltip = f"{m['category']} | conf {m['confidence']}"

        folium.PolyLine(
            m["r1"]["points"],
            color=color,
            weight=3,
            opacity=opacity,
            tooltip=f"{p1_name} | {tooltip}"
        ).add_to(m_map)

        folium.PolyLine(
            m["r2"]["points"],
            color=color,
            weight=3,
            opacity=opacity,
            dash_array="8 4",
            tooltip=f"{p2_name} | {tooltip}"
        ).add_to(m_map)

    m_map.save(path)
    print(f"✅ Map → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file1")
    ap.add_argument("file2")
    ap.add_argument("--p1-name", default="Person 1")
    ap.add_argument("--p2-name", default="Person 2")
    ap.add_argument("--radius-m", type=float, default=175)
    ap.add_argument("--path-dist-m", type=float, default=300)
    ap.add_argument("--min-overlap", type=float, default=0.25)
    ap.add_argument("--map", default="shared_drives.html")
    ap.add_argument("--csv", default="shared_drives.csv")
    args = ap.parse_args()

    p1_routes = parse_timeline(args.file1)
    p2_routes = parse_timeline(args.file2)

    matches = find_shared_drives(
        p1_routes,
        p2_routes,
        radius_m=args.radius_m,
        path_dist_m=args.path_dist_m,
        min_overlap_ratio=args.min_overlap,
    )

    print(f"{len(matches)} candidate shared trips found")

    build_csv(matches, args.p1_name, args.p2_name, args.csv)
    build_map(matches, args.p1_name, args.p2_name, args.map)


if __name__ == "__main__":
    main()