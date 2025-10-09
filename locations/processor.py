import requests
import os
from config import *
import pandas as pd

def reverse_geocode(lat, lon):
    """Return (location_name, address) using OpenStreetMap Nominatim API."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "addressdetails": 1,
        "zoom": 18,
    }
    headers = {"User-Agent": "my-geocoder-app"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Extract name and address
        name = data.get("name", None)

        address = None
        if "address" in data:
            addr = data["address"]
            # Build a concise address line (e.g., "77 Massachusetts Ave")
            address = ", ".join([
                part for part in [
                    addr.get("house_number"),
                    addr.get("road")
                ] if part
            ])

        return pd.Series([name, address])

    except Exception as e:
        print(f"Error for {lat}, {lon}: {e}")
        return pd.Series([None, None])
    

def process_data(df, weeks):
    df[["location_name", "address"]] = df.apply(
        lambda row: reverse_geocode(row["latitude"], row["longitude"]),
        axis=1
    )
    df.to_csv(os.path.join(OUTPUT_FOLDER, f'labeled_locations_{weeks[0]}.csv'))
    return df