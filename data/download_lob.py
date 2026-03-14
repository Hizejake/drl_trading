"""
LOB Data Pipeline — Parses real LOBSTER data and generates synthetic fallback.

LOBSTER (Limit Order Book System – The Efficient Reconstructor) provides 
tick-level order book snapshots from NASDAQ. This script parses the raw
LOBSTER orderbook + message CSVs into our standardized 40-column format.

Expected files in data/raw/:
  {TICKER}_{DATE}_{START}_{END}_orderbook_10.csv
  {TICKER}_{DATE}_{START}_{END}_message_10.csv

Usage:
    python data/download_lob.py                   # Parse all LOBSTER files in data/raw/
    python data/download_lob.py --ticker AAPL      # Parse specific ticker
    python data/download_lob.py --synthetic-only   # Only generate synthetic data
"""

import os
import glob
import argparse
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")


def parse_lobster_data(ticker="AAPL"):
    """
    Parse LOBSTER orderbook + message CSVs into our standardized format.
    
    LOBSTER orderbook format (40 cols, no header, 10 levels):
        ask_price_1, ask_size_1, bid_price_1, bid_size_1,
        ask_price_2, ask_size_2, bid_price_2, bid_size_2, ...
    Prices are integers: dollar price × 10000 (e.g. $585.93 → 5859300)
    
    LOBSTER message format (6 cols, no header):
        time, event_type, order_id, size, price, direction
    Time is seconds after midnight with decimal precision.
    
    Our output format (42 cols):
        timestamp, symbol,
        bid_price_1..10, ask_price_1..10, bid_size_1..10, ask_size_1..10
    """
    # Find matching files
    ob_pattern = os.path.join(DATA_DIR, f"{ticker}_*_orderbook_10.csv")
    msg_pattern = os.path.join(DATA_DIR, f"{ticker}_*_message_10.csv")
    
    ob_files = glob.glob(ob_pattern)
    msg_files = glob.glob(msg_pattern)
    
    if not ob_files:
        print(f"[LOBSTER] No orderbook file found for {ticker} in {DATA_DIR}")
        return None
    
    ob_path = ob_files[0]
    msg_path = msg_files[0] if msg_files else None
    
    print(f"[LOBSTER] Parsing {ticker}...")
    print(f"  Orderbook: {os.path.basename(ob_path)}")
    if msg_path:
        print(f"  Message:   {os.path.basename(msg_path)}")
    
    # ── Read orderbook ────────────────────────────────────────────────────
    # LOBSTER interleaves: ask_p1, ask_s1, bid_p1, bid_s1, ask_p2, ...
    lobster_cols = []
    for level in range(1, 11):
        lobster_cols.extend([
            f"ask_price_{level}", f"ask_size_{level}",
            f"bid_price_{level}", f"bid_size_{level}",
        ])
    
    df_ob = pd.read_csv(ob_path, header=None, names=lobster_cols)
    print(f"  Raw orderbook: {len(df_ob):,} rows × {len(df_ob.columns)} cols")
    
    # ── Read message file for timestamps ──────────────────────────────────
    if msg_path:
        df_msg = pd.read_csv(msg_path, header=None,
                             names=["timestamp", "event_type", "order_id", "size", "price", "direction"])
        timestamps = df_msg["timestamp"].values
    else:
        timestamps = np.arange(len(df_ob), dtype=float)
    
    # ── Convert prices: integer → dollars (÷ 10000) ──────────────────────
    for level in range(1, 11):
        df_ob[f"ask_price_{level}"] = df_ob[f"ask_price_{level}"] / 10000.0
        df_ob[f"bid_price_{level}"] = df_ob[f"bid_price_{level}"] / 10000.0
    
    # ── Filter out dummy rows (LOBSTER fills empty levels with ±9999999999)
    # A dummy is identifiable by size = 0 at level 1
    mask = (df_ob["bid_size_1"] > 0) & (df_ob["ask_size_1"] > 0)
    # Also filter extreme dummy prices
    mask &= (df_ob["bid_price_1"] > 0) & (df_ob["ask_price_1"] < 999999)
    
    df_ob = df_ob[mask].reset_index(drop=True)
    timestamps = timestamps[mask.values] if len(timestamps) == len(mask) else timestamps[:len(df_ob)]
    
    # ── Reorder columns to our standard format ────────────────────────────
    # Our format: bid_price_1..10, ask_price_1..10, bid_size_1..10, ask_size_1..10
    out_cols = (
        [f"bid_price_{i}" for i in range(1, 11)] +
        [f"ask_price_{i}" for i in range(1, 11)] +
        [f"bid_size_{i}" for i in range(1, 11)] +
        [f"ask_size_{i}" for i in range(1, 11)]
    )
    
    df_out = pd.DataFrame()
    df_out["timestamp"] = timestamps[:len(df_ob)]
    df_out["symbol"] = ticker
    for col in out_cols:
        df_out[col] = df_ob[col].values
    
    # ── Save ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(DATA_DIR, f"lobster_{ticker.lower()}_10_level.csv")
    df_out.to_csv(csv_path, index=False)
    
    # Print summary
    mid = (df_out["bid_price_1"] + df_out["ask_price_1"]) / 2
    spread = df_out["ask_price_1"] - df_out["bid_price_1"]
    print(f"  Saved: {csv_path}")
    print(f"  Stats: {len(df_out):,} ticks, mid=${mid.mean():.2f}, "
          f"spread=${spread.mean()*100:.2f}¢, spread_bps={spread.mean()/mid.mean()*10000:.1f}bps")
    
    return csv_path


def generate_synthetic_lob(ticks=5000):
    """Generate synthetic LOB data for testing."""
    print(f"[SYNTHETIC] Generating {ticks}-tick synthetic LOB data...")

    base_price = 150.00
    data = {}
    price_walk = np.cumsum(np.random.normal(0, 0.02, ticks))
    
    for level in range(1, 11):
        bid_prices = base_price + price_walk - (level * 0.01)
        ask_prices = base_price + price_walk + (level * 0.01)
        data[f"bid_price_{level}"] = np.round(bid_prices + np.random.normal(0, 0.003, ticks), 4)
        data[f"ask_price_{level}"] = np.round(ask_prices + np.random.normal(0, 0.003, ticks), 4)
        data[f"bid_size_{level}"] = np.random.randint(50 * level, 500 * level, size=ticks)
        data[f"ask_size_{level}"] = np.random.randint(50 * level, 500 * level, size=ticks)

    df = pd.DataFrame(data)
    timestamps = pd.date_range(start="2024-01-01 09:30:00", periods=ticks, freq="100ms")
    df.insert(0, "timestamp", timestamps)
    df.insert(1, "symbol", "SYNTH")
    
    csv_path = os.path.join(DATA_DIR, "synthetic_lob_10_level.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SYNTHETIC] Saved {ticks} ticks to {csv_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse LOB data")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--ticker", default=None, help="Parse specific ticker (default: all found)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    if args.synthetic_only:
        generate_synthetic_lob()
    else:
        if args.ticker:
            parse_lobster_data(args.ticker)
        else:
            # Auto-detect all LOBSTER tickers in data/raw/
            ob_files = glob.glob(os.path.join(DATA_DIR, "*_orderbook_10.csv"))
            tickers = set()
            for f in ob_files:
                basename = os.path.basename(f)
                ticker = basename.split("_")[0]
                tickers.add(ticker)
            
            if tickers:
                print(f"[LOBSTER] Found tickers: {sorted(tickers)}")
                for ticker in sorted(tickers):
                    try:
                        parse_lobster_data(ticker)
                    except Exception as e:
                        print(f"[ERROR] Failed to parse {ticker}: {e}")
            else:
                print("[LOBSTER] No LOBSTER files found in data/raw/. Generating synthetic data...")
                generate_synthetic_lob()
        
        # Always regenerate synthetic as well
        generate_synthetic_lob()
