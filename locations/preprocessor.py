import os
import json
import pandas as pd
from config import *


def load_json_file(file_path):
    """Load a single JSON file into a pandas DataFrame."""
    with open(file_path, "r") as f:
        data = json.load(f)
    print(file_path)
    map_df = pd.json_normalize(data["semanticSegments"])
    map_df = map_df[map_df["visit.topCandidate.placeId"].notnull()]

    subfolder_name = os.path.basename(os.path.dirname(file_path))
    map_df["person"] = subfolder_name
    return map_df


def get_json_files(base_folder, subfolders):
    """
    Return a list of JSON file paths from the specified subfolders.
    If subfolders == ['all'], include all nested subfolders.
    """
    json_files = []

    # Determine which folders to search
    if subfolders == ["all"]:
        folders_to_walk = [base_folder]
    else:
        folders_to_walk = [
            os.path.join(base_folder, s)
            for s in subfolders
            if os.path.exists(os.path.join(base_folder, s))
        ]

    # Collect JSON files
    for folder in folders_to_walk:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))

    return json_files


def load_jsons_from_folders(base_folder, subfolders):
    json_files = get_json_files(base_folder, subfolders)
    if not json_files:
        print("No JSON files found.")
        return pd.DataFrame()

    dfs = [load_json_file(path) for path in json_files]
    return pd.concat(dfs, ignore_index=True)


def clean_visits_df(df):
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
    df['startTime'] = pd.to_datetime(df['startTime'], errors = 'coerce')
    df['endTime'] = pd.to_datetime(df['endTime'], errors = 'coerce')
    # df['week'] = df['startTime'].dt.isocalendar().week

    return df


def preprocess_data():
    df = load_jsons_from_folders(DATA_FOLDER, WEEKS)
    clean_df = clean_visits_df(df)
    return clean_df
