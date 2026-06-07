import pandas as pd
import requests

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
POLYGON_API_KEY = "OgzZiXrrqfqbPVOrDVSWVJsOtbyknXWi"
TICKER = "AAPL"
START_DATE = "2024-01-01"
END_DATE = "2024-01-02"  # Extend by one day to catch edge cases

# ── FETCH DATA ────────────────────────────────────────────────────────────────
url = f"https://api.massive.io/v2/reference/news" #rebranded to massive
params = {
    "ticker": TICKER,
    "published_utc.gte": START_DATE,
    "published_utc.lte": END_DATE,
    "order": "asc",  # Sort chronologically right from the API
    "limit": 1000,  # Max allowable items per request
    "apiKey": POLYGON_API_KEY,
}

all_results = []
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    all_results.extend(data.get("results", []))

    # Polygon uses a 'next_url' pagination token if results exceed the limit
    while "next_url" in data:
        next_url = f"{data['next_url']}&apiKey={POLYGON_API_KEY}"
        response = requests.get(next_url)
        if response.status_code == 200:
            data = response.json()
            all_results.extend(data.get("results", []))
        else:
            break

    # Convert to DataFrame
    df_news = pd.DataFrame(all_results)

    if not df_news.empty:
        # Save offline dataset
        output_path = f"polygon_{TICKER.lower()}_news.csv"
        df_news.to_csv(output_path, index=False)
        print(
            f"Successfully saved {len(df_news)} AAPL news articles to {output_path}"
        )
    else:
        print("No news found for the given parameters.")
else:
    print(f"Error fetching from Polygon: {response.status_code}, {response.text}")