from maps_utils import *
import requests


def reverse_geocode(lat, lon):
    """Call OpenStreetMap Nominatim API to get name or address."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1,
        "zoom": 18,   # higher zoom gives more specific results
    }
    headers = {"User-Agent": "my-geocoder-app"} 

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Prefer business/POI name if available
        if "name" in data:
            return data["name"]
        elif "display_name" in data:
            return data["display_name"]
        else:
            return None
    except Exception as e:
        print(f"Error for {lat}, {lon}: {e}")
        return None


def map_common_clusters(map_folder, p):
    # df = retrieve_json(map_folder)
    # cleaned_df = clean_visits_df(df)
    path = os.path.join(map_folder,f'locations_with_names_{p}.csv' )
    df = pd.read_csv(path)
    df['startTime'] = pd.to_datetime(df['startTime'])
    df['endTime'] = pd.to_datetime(df['endTime'])
    df['week'] = df['startTime'].dt.isocalendar().week
    df, recurring_clusters = cluster_weekly_locations(df, 0.5)
    # cluster_weekday = recurring_clusters_by_weekday(df)
    create_weekly_locations_map(df, recurring_clusters, p)

def map_weekly_clusters(map_folder, p):
    path = os.path.join(map_folder,f'locations_with_names_{p}.csv' )
    df = pd.read_csv(path)    
    df['startTime'] = pd.to_datetime(df['startTime'])
    df['endTime'] = pd.to_datetime(df['endTime'])
    df['week'] = df['startTime'].dt.isocalendar().week
    df, _ = cluster_weekly_locations(df, 0.5)
    cluster_weekday = recurring_clusters_by_weekday(df)
    create_map_weekday(df, cluster_weekday, p)

def label_visits(map_folder, p):
    df = retrieve_json(map_folder)
    df = clean_visits_df(df)
    df["location_name"] = df.apply(lambda row: reverse_geocode(row["latitude"], row["longitude"]), axis=1)
    path = f"/Users/esmerubinstein/Desktop/ICLL/AICaring/AICaring/maps_data/{p}"
    output_file = f"locations_with_names_{p}.csv"
    output_path = os.path.join(path, output_file)
    df.to_csv(output_path)


def retrieve_cluster_locations(map_folder, p):
    path = os.path.join(map_folder,f'locations_with_names_{p}.csv' )
    df = pd.read_csv(path)
    df['location_name'] = df['location_name'].fillna("No Identified Location")
    df, recurring_clusters = cluster_weekly_locations(df, 0.5)
    cluster_locations = get_cluster_locations(df, recurring_clusters)
    with open(os.path.join(map_folder, f"cluster_locations_{p}.json"), "w") as f:
        json.dump(cluster_locations, f, indent=4)



map_folder = "/Users/esmerubinstein/Desktop/ICLL/AICaring/AICaring/maps_data/J/"
map_weekly_clusters(map_folder, "J")
