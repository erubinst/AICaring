
from google import genai
import pandas as pd
import time
import pickle
import json
from config import *

client = genai.Client(api_key="AIzaSyComIKziB9u4r95SVGFIYk6RcPpiFmY-Jg")


# Prompt construction
def build_prompt(batch, guided=False):
    """Builds a batch prompt for Gemini classification."""
    if guided:
        category_prompt = (
            "Choose the best matching category from this list:\n"
            "['grocery store','church', 'pharmacy', 'doctor office', 'laundromat', 'community center',"
            "'restaurant', 'bank', 'hospital', 'gas station', 'school', 'gym', 'retail store', 'other', 'post office'].\n"
        )
    else:
        category_prompt = (
            "For each location below, infer a short, specific category describing the type of business or place. "
            "Use a general noun phrase like 'pharmacy', 'hardware store', 'coffee shop', 'university', etc."
        )

    batch_prompt = "\n".join([
        f"{i+1}. Name: {row['location_name']}"
        for i, (_, row) in enumerate(batch.iterrows())
    ])

    return f"""
    {category_prompt}

    Provide your answer **only** as a numbered list (1–{len(batch)}), one category per line.
    Do not include any introduction, explanation, or extra text — only the numbered list.

    {batch_prompt}
    """


# Call gemini
def query_gemini(prompt, model_name="gemini-2.5-flash"):
    """Sends prompt to Gemini and returns text output."""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    print(response.text)
    return response.text.strip()


# Parse gemini output for batch
def parse_categories(response_text, expected_count):
    """Parses Gemini output into a clean list of category strings."""
    categories = []
    for line in response_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "." in line:
            line = line.split(".", 1)[-1].strip()
        categories.append(line)

    # Pad with 'unknown' if Gemini returns fewer lines than expected
    while len(categories) < expected_count:
        categories.append("unknown")

    return categories[:expected_count]


# Categorize a particular batch
def categorize_batch(df_batch, cache, guided=False):
    """Categorizes a single batch, using cache when available."""
    uncached_rows = []
    results = []

    for _, row in df_batch.iterrows():
        key = (str(row["location_name"]).strip().lower(), str(row["address"]).strip().lower())
        if key in cache:
            results.append(cache[key])
        else:
            uncached_rows.append((row, key))

    # If all cached, return
    if not uncached_rows:
        return results

    # Build and send prompt
    prompt = build_prompt(df_batch, guided=guided)
    try:
        response_text = query_gemini(prompt)
        batch_categories = parse_categories(response_text, len(df_batch))

        # Update cache
        for (row, key), category in zip(uncached_rows, batch_categories):
            cache[key] = category
            results.append(category)

    except Exception as e:
        print(f"Error processing batch: {e}")
        for (row, key) in uncached_rows:
            cache[key] = "unknown"
            results.append("unknown")

    return results


# Categorization pipeline
def categorize_locations(df, batch_size=10, guided=False, sleep=1.0, cache_file=None):
    """
    Categorize locations using Gemini in batches, with caching and optional persistence.
    Returns (updated_df, updated_cache)
    """
    # Load existing cache if available
    cache = {}
    if cache_file:
        try:
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
            print(f"Loaded {len(cache)} cached entries.")
        except FileNotFoundError:
            print("No existing cache file found — starting fresh.")

    all_categories = []

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        print(f"Processing rows {start}–{start + len(batch) - 1}...")

        batch_results = categorize_batch(batch, cache, guided=guided)
        all_categories.extend(batch_results)

        time.sleep(sleep)  # prevent rate limiting

    df = df.copy()
    df["category"] = all_categories

    # Save cache if requested
    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)
        print(f"Cache saved with {len(cache)} entries.")

    return df, cache


def save_locations_by_category(df, output_path):
    """
    Groups a DataFrame by 'category' and saves a JSON file where
    each category contains a unique list of {location_name, address} entries.
    Duplicate (location_name, address) pairs are removed.
    """

    # Group and build structure
    grouped = (
        df.groupby('category')
        .apply(lambda g: [
            {'location_name': row.location_name, 'address': row.address}
            for _, row in g.iterrows()
        ])
        .to_dict()
    )

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(grouped, f, indent=4, ensure_ascii=False)


def categorize_data(weeks):

    data_path = f'{ABSOLUTE_PATH}/labeled_locations/labeled_locations_{weeks[0]}.csv'
    df = pd.read_csv(data_path)
    df_labeled_location = df[~df['location_name'].isnull()]
    df_unique = df_labeled_location.drop_duplicates(subset=['location_name', 'address'])
    df_with_categories, cache = categorize_locations(
        df_unique,
        batch_size=10,
        guided=True,        # or True if you want to constrain categories
        cache_file="category_cache.pkl",
    )
    df_with_categories.to_csv(f'{ABSOLUTE_PATH}/categorized_locations/categorized_locations_{weeks[0]}.csv')
    categorization_output_path = f'{ABSOLUTE_PATH}/categorized_locations/categorized_locations_{weeks[0]}.json'
    save_locations_by_category(df_with_categories, categorization_output_path)
