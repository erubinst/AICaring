import json
import numpy as np
import pandas as pd
import folium

# within 100 m could have piggybacked

def parse_latlon(s):
    """
    Convert '40.465173°, -79.923901°' -> (lat, lon)
    """
    lat, lon = s.replace("°", "").split(",")
    return float(lat.strip()), float(lon.strip())


def haversine_meters_vec(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine distance in meters
    """
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))


def extract_points(json_path, person_id):
    """
    Extract GPS points from Google Maps Timeline JSON
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    rows = []

    for seg in data.get("semanticSegments", []):
        if "timelinePath" in seg:
            for p in seg["timelinePath"]:
                lat, lon = parse_latlon(p["point"])
                rows.append({
                    "person_id": person_id,
                    "timestamp": pd.to_datetime(p["time"]),
                    "lat": lat,
                    "lon": lon
                })

        if "visit" in seg:
            loc = seg["visit"]["topCandidate"]["placeLocation"]["latLng"]
            lat, lon = parse_latlon(loc)
            rows.append({
                "person_id": person_id,
                "timestamp": pd.to_datetime(seg["startTime"]),
                "lat": lat,
                "lon": lon
            })

    return pd.DataFrame(rows)


def downsample(df, minutes=5):
    """
    One point per person per time window
    """
    return (
        df.sort_values("timestamp")
          .set_index("timestamp")
          .groupby("person_id", group_keys=False)
          .resample(f"{minutes}min")
          .first()
          .dropna()
          .reset_index()
    )


def deduplicate_points(df, radius_m=100):
    """
    Collapse points within radius_m into a single representative point.
    Greedy spatial deduplication.
    """
    kept = []

    for _, row in df.iterrows():
        if not kept:
            kept.append(row)
            continue

        kept_df = pd.DataFrame(kept)

        dists = haversine_meters_vec(
            row.lat, row.lon,
            kept_df.lat.values,
            kept_df.lon.values
        )

        if np.all(dists > radius_m):
            kept.append(row)

    return pd.DataFrame(kept)


def find_shared_locations(p1, p2, radius_m=100):
    """
    Find pairs of locations (one from each person) that are within radius_m.
    Returns a DataFrame with columns: p1_lat, p1_lon, p2_lat, p2_lon
    """
    shared = []

    for _, r1 in p1.iterrows():
        dists = haversine_meters_vec(
            r1.lat, r1.lon,
            p2.lat.values,
            p2.lon.values
        )

        close_idxs = np.where(dists <= radius_m)[0]

        for idx in close_idxs:
            r2 = p2.iloc[idx]
            shared.append({
                "p1_lat": r1.lat,
                "p1_lon": r1.lon,
                "p2_lat": r2.lat,
                "p2_lon": r2.lon
            })

    return pd.DataFrame(shared)


def cluster_midpoints(latitudes, longitudes, eps_m=200):
    """
    Simple spatial clustering (DBSCAN-style) on midpoints using a distance threshold eps_m (meters).
    Returns (labels, cluster_centers), where:
      - labels: array of cluster ids for each point
      - cluster_centers: list of (center_lat, center_lon) per cluster id
    """
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)
    n = len(latitudes)
    if n == 0:
        return np.array([]), []

    labels = np.full(n, -1, dtype=int)
    cluster_centers = []
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue  # already assigned to a cluster

        # Start a new cluster
        labels[i] = cluster_id
        queue = [i]

        # BFS / flood-fill over neighbors within eps_m
        while queue:
            idx = queue.pop()
            dists = haversine_meters_vec(
                latitudes[idx],
                longitudes[idx],
                latitudes,
                longitudes
            )
            neighbors = np.where((dists <= eps_m) & (labels == -1))[0]

            for nb in neighbors:
                labels[nb] = cluster_id
                queue.append(nb)

        # Compute cluster center for this cluster
        cluster_mask = labels == cluster_id
        center_lat = latitudes[cluster_mask].mean()
        center_lon = longitudes[cluster_mask].mean()
        cluster_centers.append((center_lat, center_lon))

        cluster_id += 1

    return labels, cluster_centers


def get_route_to_cluster(df, center_lat, center_lon, region_radius_m=100):
    """
    Given a time-ordered df with columns ['lat', 'lon'],
    return a list of (lat, lon) representing the route INTO the cluster:
      - from the last point outside the region
      - to the first point inside the region
    If the person never enters the region, return [].
    """
    if df.empty:
        return []

    # Distances from every point to the cluster center
    dists = haversine_meters_vec(
        center_lat,
        center_lon,
        df["lat"].values,
        df["lon"].values
    )

    in_cluster_idxs = np.where(dists <= region_radius_m)[0]
    if len(in_cluster_idxs) == 0:
        return []

    # First time this person enters the region
    entry_idx = int(in_cluster_idxs[0])

    # Walk backward to find last index outside the region
    start_idx = 0
    for i in range(entry_idx - 1, -1, -1):
        if dists[i] > region_radius_m:
            start_idx = i
            break

    # Extract coordinates from start_idx to entry_idx (inclusive)
    sub = df.loc[start_idx:entry_idx]
    coords = list(zip(sub["lat"], sub["lon"]))
    return coords


def visualize_shared(df1, df2, shared_df, output_html="shared_locations.html"):
    """
    Visualize:
      - Person 1 locations in red
      - Person 2 locations in blue
      - Shared locations in purple
      - Clustered shared regions as a single black circle per cluster
      - For each cluster, draw:
          - thick red line: Person 1's route into that cluster
          - thick blue line: Person 2's route into that cluster
    """
    if df1.empty and df2.empty:
        print("No points to visualize.")
        return

    # Sort by time so "routes" make sense
    df1_sorted = df1.sort_values("timestamp").reset_index(drop=True)
    df2_sorted = df2.sort_values("timestamp").reset_index(drop=True)

    # Center the map on all points
    all_points = pd.concat(
        [
            df1_sorted[["lat", "lon"]].assign(person="p1"),
            df2_sorted[["lat", "lon"]].assign(person="p2")
        ],
        ignore_index=True
    )
    center_lat = all_points["lat"].mean()
    center_lon = all_points["lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Only show clusters and routes into clusters
    # No purple shared dots

    # Only show clusters and routes into clusters
    if not shared_df.empty:
        mid_lats = (shared_df["p1_lat"] + shared_df["p2_lat"]) / 2.0
        mid_lons = (shared_df["p1_lon"] + shared_df["p2_lon"]) / 2.0

        labels, cluster_centers = cluster_midpoints(
            mid_lats.values,
            mid_lons.values,
            eps_m=500
        )

        region_radius_m = 500

        for cluster_id, (center_lat, center_lon) in enumerate(cluster_centers):
            folium.Circle(
                location=[center_lat, center_lon],
                radius=region_radius_m,
                color="black",
                weight=2,
                fill=False
            ).add_to(m)

            # Draw route for Person 1 into this cluster (thick red line)
            route1_coords = get_route_to_cluster(
                df1_sorted,
                center_lat,
                center_lon,
                region_radius_m=region_radius_m
            )
            if route1_coords:
                folium.PolyLine(
                    locations=route1_coords,
                    color="red",
                    weight=6,
                    opacity=0.8
                ).add_to(m)
                for lat, lon in route1_coords:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6,
                        color="red",
                        fill=True,
                        fill_opacity=0.9,
                        popup="Route to cluster (Person 1)"
                    ).add_to(m)

            # Draw route for Person 2 into this cluster (thick blue line)
            route2_coords = get_route_to_cluster(
                df2_sorted,
                center_lat,
                center_lon,
                region_radius_m=region_radius_m
            )
            if route2_coords:
                folium.PolyLine(
                    locations=route2_coords,
                    color="blue",
                    weight=6,
                    opacity=0.8
                ).add_to(m)
                for lat, lon in route2_coords:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=6,
                        color="blue",
                        fill=True,
                        fill_opacity=0.9,
                        popup="Route to cluster (Person 2)"
                    ).add_to(m)

    # (Remove duplicate cluster drawing logic)

    m.save(output_html)
    print(f"Map saved to {output_html}")


def main():
    df1 = extract_points("person1.json", person_id=1)
    df1 = downsample(df1)

    df2 = extract_points("person2.json", person_id=2)
    df2 = downsample(df2)

    # Deduplicate per person
    df1_unique = deduplicate_points(df1, radius_m=100)
    df2_unique = deduplicate_points(df2, radius_m=100)

    print(f"Person A points: {len(df1)} → {len(df1_unique)}")
    print(f"Person B points: {len(df2)} → {len(df2_unique)}")

    # You can use 100 here if you want "within 100m could have piggybacked"
    shared = find_shared_locations(df1_unique, df2_unique, radius_m=100)

    print(f"Shared locations: {len(shared)}")

    # Pass both unique point sets + the shared pairs to visualization
    visualize_shared(df1_unique, df2_unique, shared)


if __name__ == "__main__":
    main()