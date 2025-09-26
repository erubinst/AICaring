import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from datetime import datetime
import matplotlib.pyplot as plt
import folium
import os
import json


def retrieve_json(folder):
    map_folder = folder
    # get all files in folder
    map_files = [f for f in os.listdir(map_folder) if f.endswith(".json")]
    dfs = []
    for map_file in map_files:
        with open(map_folder + map_file, "r") as f:
            data = json.load(f)
        map_df = pd.json_normalize(data["semanticSegments"])
        map_df = map_df[map_df["visit.topCandidate.placeId"].notnull()]
        dfs.append(map_df)
    all_maps_df = pd.concat(dfs, ignore_index=True)
    all_maps_df = all_maps_df.reset_index(drop=True)

    return all_maps_df


def clean_df(df):
    visit_cols = [col for col in df.columns if col.startswith("visit.")]
    relevant_cols = ["startTime", "endTime"] + visit_cols
    df = df[relevant_cols]

    # expand lat lng into two separate cols from list
    latlng_col = "visit.topCandidate.placeLocation.latLng"

    # Split on the comma
    df[["latitude", "longitude"]] = (
        df[latlng_col]
        .str.replace("°", "", regex=False)   # remove the degree symbol
        .str.split(",", expand=True)         # split into two columns
    )

    df["latitude"] = pd.to_numeric(df["latitude"].str.strip())
    df["longitude"] = pd.to_numeric(df["longitude"].str.strip())
    df['startTime'] = pd.to_datetime(df['startTime'])
    df['endTime'] = pd.to_datetime(df['endTime'])
    df['week'] = df['startTime'].dt.isocalendar().week

    return df

def extract_temporal_patterns(df, recurring_clusters):
    # Add day of week
    df['day_of_week'] = df['startTime'].dt.dayofweek  # 0=Mon, 6=Sun

    # Keep only recurring clusters
    recurring_df = df[df['cluster'].isin(recurring_clusters.index)]

    # Count visits per cluster per day of week
    dow_summary = recurring_df.groupby(['cluster', 'day_of_week']).size().unstack(fill_value=0)
    print(dow_summary)


def cluster_locations(df):
    # Convert lat/lon to radians for Haversine
    coords = df[['latitude', 'longitude']].to_numpy()
    coords_rad = np.radians(coords)

    # DBSCAN with Haversine distance (eps in radians)
    kms_per_radian = 6371.0088
    epsilon = 0.5 / kms_per_radian  # e.g., 500m radius

    db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine')
    df['cluster'] = db.fit_predict(coords_rad)

    # Remove noise points (cluster = -1)
    df = df[df['cluster'] != -1]

    weekly_visits = df.groupby(['cluster', 'week']).size().unstack(fill_value=0)
    print(weekly_visits)
    recurring_clusters = weekly_visits[weekly_visits.gt(0).sum(axis=1) >= 3]
    print(recurring_clusters)
    return df, recurring_clusters


def create_map(df, recurring_clusters, p):
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=13)

    # Plot recurring clusters
    for cluster_id in recurring_clusters.index:
        cluster_points = df[df['cluster'] == cluster_id][['latitude', 'longitude']]
        center = cluster_points.mean()
        folium.Circle(
            location=[center.latitude, center.longitude],
            radius=700,  # radius in meters (so 1 km here)
            color='blue',
            fill=True,
            fill_opacity=0.2,
            popup=f'Cluster {cluster_id}'
        ).add_to(m)

    m.save(f"recurring_locations_{p}.html")

def recurring_clusters_by_weekday(df, min_weeks=3):
    """
    Returns clusters with labels listing all weekdays where the cluster is visited
    in at least `min_weeks` separate weeks.
    """
    df['day_of_week'] = df['startTime'].dt.dayofweek  # 0=Mon, 6=Sun

    # Count unique weeks per cluster per weekday
    cluster_weekday = (
        df.groupby(['cluster', 'day_of_week'])['week']
        .nunique()
        .reset_index(name='weeks_visited')
    )

    # Keep only weekdays with >= min_weeks visits
    cluster_weekday = cluster_weekday[cluster_weekday['weeks_visited'] >= min_weeks]

    # Total weeks in dataset
    total_weeks = df['week'].nunique()

    # Map weekday numbers to names
    weekday_names = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}

    # Aggregate labels per cluster
    labels = (
        cluster_weekday.groupby('cluster')
        .apply(lambda x: ", ".join([f"{weekday_names[d]}: {w}/{total_weeks}" 
                                    for d, w in zip(x['day_of_week'], x['weeks_visited'])]))
        .reset_index(name='label')
    )

    return labels


def create_map_weekday(df, recurring_clusters, P):
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=13)
    plotted_clusters = set()

    for _, row in recurring_clusters.iterrows():
        cluster_id = row['cluster']
        if cluster_id in plotted_clusters:
            continue  # only plot each cluster once
        plotted_clusters.add(cluster_id)

        cluster_points = df[df['cluster'] == cluster_id][['latitude', 'longitude']]
        center = cluster_points.mean()

        folium.Circle(
            location=[center.latitude, center.longitude],
            radius=700,
            color='blue',
            fill=True,
            fill_opacity=0.4,
            popup=f'Cluster {cluster_id}: {row["label"]}'
        ).add_to(m)

    m.save(f"recurring_locations_{P}_weekday.html")


def analyze_data(map_folder, p):
    df = retrieve_json(map_folder)
    cleaned_df = clean_df(df)
    df, recurring_clusters = cluster_locations(cleaned_df)
    cluster_weekday = recurring_clusters_by_weekday(df)
    create_map_weekday(df, cluster_weekday, p)
    # extract_temporal_patterns(df, recurring_clusters)
    # create_map(cleaned_df, recurring_clusters, p)

# given map folder, loop through all of them and create df
map_folder = "/Users/esmerubinstein/Desktop/ICLL/AICaring/AICaring/maps_data/J/"
# analyze_data(map_folder, "J")
map_folder = "/Users/esmerubinstein/Desktop/ICLL/AICaring/AICaring/maps_data/C/"
# analyze_data(map_folder, "C")
map_folder = "/Users/esmerubinstein/Desktop/ICLL/AICaring/AICaring/maps_data/R/"
# analyze_data(map_folder, "R")
map_folder = "/Users/esmerubinstein/Desktop/ICLL/AICaring/AICaring/maps_data/A/"
analyze_data(map_folder, "A")