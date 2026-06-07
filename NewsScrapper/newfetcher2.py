"""
Fetch AAPL news from Polygon.io for the exact date range in the LOB CSV.

Reads data/raw/lobster_aapl_10_level.csv, extracts its first/last timestamps,
then fetches matching Polygon news. Output is consumed by the swarm cell in
drl_trading_pipeline.ipynb.

Output : data/raw/polygon_aapl_news.csv

Usage:
    python NewsScrapper/newfetcher2.py                 # auto-detects dates from LOB CSV
    python NewsScrapper/newfetcher2.py --ticker MSFT   # different stock
    python NewsScrapper/newfetcher2.py --start 2024-01-01 --end 2024-06-30
"""

import os
import sys
import time
import argparse

import requests
import pandas as pd

# ── CONFIG ─────────────────────────────────────────────────────────────────────
POLYGON_API_KEY = "OgzZiXrrqfqbPVOrDVSWVJsOtbyknXWi"
TICKER     = "AAPL"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
NEWS_URL   = "https://api.polygon.io/v2/reference/news"


# ── HELPERS ────────────────────────────────────────────────────────────────────

def dates_from_lob(ticker: str) -> tuple[str, str] | None:
    """Read LOB CSV and return (start_date, end_date) strings."""
    lob_path = os.path.join(OUTPUT_DIR, f"lobster_{ticker.lower()}_10_level.csv")
    if not os.path.exists(lob_path):
        return None
    df = pd.read_csv(lob_path, usecols=["timestamp"])
    ts    = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    start = ts.min().strftime("%Y-%m-%d")
    end   = ts.max().strftime("%Y-%m-%d")
    print(f"LOB CSV: {lob_path}  ({len(df):,} ticks  {start} → {end})")
    return start, end


def fetch_all_news(ticker: str, start: str, end: str, api_key: str) -> list[dict]:
    """Fetch all pages of Polygon news for ticker in [start, end]."""
    articles, url = [], NEWS_URL
    params = {
        "ticker":              ticker,
        "published_utc.gte":   f"{start}T00:00:00Z",
        "published_utc.lte":   f"{end}T23:59:59Z",
        "order":               "asc",
        "limit":               1000,
        "apiKey":              api_key,
    }

    while url:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if data.get("status") not in ("OK", "DELAYED"):
            print(f"  [warn] Unexpected status: {data.get('status')}")
            break

        batch = data.get("results", [])
        articles.extend(batch)
        print(f"  Page fetched: {len(batch)} articles  (total so far: {len(articles)})")

        # Pagination: next_url comes without apiKey
        next_url = data.get("next_url")
        if next_url:
            url    = next_url
            params = {"apiKey": api_key}          # key goes as param, not in base url
            time.sleep(0.5)                        # stay within free-tier rate limit
        else:
            break

    return articles


def best_text(row: pd.Series) -> str:
    for col in ("description", "title"):
        val = row.get(col, "")
        if isinstance(val, str) and len(val.strip()) > 20:
            return val.strip()
    return str(row.get("title", "")).strip()


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Fetch Polygon news matching LOB dates")
    ap.add_argument("--ticker",  default=TICKER)
    ap.add_argument("--start",   default=None, help="YYYY-MM-DD  (auto from LOB CSV)")
    ap.add_argument("--end",     default=None, help="YYYY-MM-DD  (auto from LOB CSV)")
    ap.add_argument("--api-key", default=POLYGON_API_KEY)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR,
                               f"polygon_{args.ticker.lower()}_news.csv")

    # ── Resolve date range ────────────────────────────────────────────────────
    if args.start and args.end:
        start, end = args.start, args.end
    else:
        result = dates_from_lob(args.ticker)
        if result is None:
            print(f"[ERROR] LOB CSV not found. Run lob_fetcher.py first, "
                  f"or pass --start / --end manually.")
            sys.exit(1)
        start, end = result

    print(f"\nPolygon News Fetcher")
    print(f"  Ticker : {args.ticker}")
    print(f"  Period : {start} → {end}")
    print(f"  Output : {output_path}\n")

    # ── Fetch ─────────────────────────────────────────────────────────────────
    try:
        articles = fetch_all_news(args.ticker, start, end, args.api_key)
    except requests.HTTPError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not articles:
        print("[WARN] No articles returned. The free tier may have limited history.")
        print("       The notebook will fall back to synthetic news events.")
        pd.DataFrame().to_csv(output_path, index=False)
        sys.exit(0)

    # ── Clean ─────────────────────────────────────────────────────────────────
    df = pd.DataFrame(articles)

    if "published_utc" in df.columns:
        df["datetime_utc"] = pd.to_datetime(df["published_utc"], utc=True)
        df = df.sort_values("datetime_utc").reset_index(drop=True)

    df["text"] = df.apply(best_text, axis=1)
    df = df[df["text"].str.len() > 20].reset_index(drop=True)

    if "title" in df.columns:
        df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    keep = [c for c in ["published_utc", "datetime_utc", "text", "title",
                         "description", "author"] if c in df.columns]
    df[keep].to_csv(output_path, index=False)

    print(f"\n✅  Saved {len(df)} articles → {output_path}")
    if not df.empty:
        print(f"    Range  : {df['datetime_utc'].min()} → {df['datetime_utc'].max()}")
        print(f"    Sample : {df['text'].iloc[0][:100]}")

    print(f"\nNext — open drl_trading_pipeline.ipynb and run the swarm cell.")
    print(f"It will auto-load headlines from: {output_path}")


if __name__ == "__main__":
    main()
