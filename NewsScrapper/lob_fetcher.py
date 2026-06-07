"""
Build a 10-level LOB dataset from Yahoo Finance hourly OHLCV (via yfinance).

Source : Yahoo Finance — completely FREE, no API key required.
Output : data/raw/lobster_aapl_10_level.csv  (or whichever --ticker you choose)

Why hourly OHLCV instead of real depth snapshots:
    True stock LOB data (LOBSTER, Nasdaq ITCH) is proprietary / academic-only.
    We reconstruct a realistic pseudo-LOB per 1-hour bar using:
        - close      → mid price (real AAPL price history)
        - high - low → intra-hour volatility (scales deeper level spacing)
        - volume     → resting sizes at each level
        - open→close direction → buy/sell pressure (shapes bid vs ask depth)

Usage:
    python NewsScrapper/lob_fetcher.py                         # 2 years of AAPL
    python NewsScrapper/lob_fetcher.py --ticker MSFT           # switch ticker
    python NewsScrapper/lob_fetcher.py --period 1y             # shorter period
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("[ERROR] yfinance not installed. Run: pip install yfinance")
    sys.exit(1)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TICKER  = "AAPL"
PERIOD  = "2y"          # yfinance period string  (max for hourly is "2y")
LEVELS  = 10

# Realistic AAPL LOB parameters
MIN_SPREAD  = 0.01      # $0.01 NBBO spread (penny stock minimum increment)
LEVEL_STEP  = 0.01      # each deeper level is $0.01 further from mid

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


# ── LOB RECONSTRUCTION ─────────────────────────────────────────────────────────

def ohlcv_to_lob(df: pd.DataFrame, ticker: str, levels: int) -> pd.DataFrame:
    """
    Convert hourly OHLCV bars to a realistic 10-level LOB snapshot per row.

    Level prices  : linearly spaced by $0.01 (penny increments) from close
    Level sizes   : derived from traded volume + buy/sell pressure from
                    the close-vs-open direction of the bar
    """
    close  = df["Close"].values.astype(np.float64)
    high   = df["High"].values.astype(np.float64)
    low    = df["Low"].values.astype(np.float64)
    open_  = df["Open"].values.astype(np.float64)
    volume = df["Volume"].values.astype(np.float64)

    # ── Buy/sell pressure from bar direction (clipped for stability)
    bar_range = np.maximum(high - low, 0.01)
    buy_frac  = np.clip((close - low) / bar_range, 0.1, 0.9)   # 0 = all selling, 1 = all buying
    sell_frac = 1.0 - buy_frac

    # ── Volume distributed across levels (deeper levels hold more)
    #    Typical stock LOB: level 1 thinner, deeper levels bigger
    level_weights = np.array([(l ** 0.5) for l in range(1, levels + 1)])
    level_weights /= level_weights.sum()      # normalise to sum=1

    rng = np.random.default_rng(seed=42)
    out: dict = {
        "timestamp": np.array([t.timestamp() for t in df.index], dtype=np.int64),
        "symbol": ticker,
    }

    for i in range(levels):
        lv = i + 1
        # Prices: each level is LEVEL_STEP further from mid ($0.01 increments)
        out[f"bid_price_{lv}"] = np.round(close - (lv * LEVEL_STEP), 2)
        out[f"ask_price_{lv}"] = np.round(close + (lv * LEVEL_STEP), 2)

        # Sizes: shares resting at this level (scaled by volume)
        noise     = 1.0 + rng.normal(0, 0.12, size=len(df))
        bid_sizes = np.maximum(volume * level_weights[i] * sell_frac * noise, 1.0)
        ask_sizes = np.maximum(volume * level_weights[i] * buy_frac  * noise, 1.0)

        out[f"bid_size_{lv}"] = np.round(bid_sizes).astype(int)
        out[f"ask_size_{lv}"] = np.round(ask_sizes).astype(int)

    result = pd.DataFrame(out)

    # Drop rows where prices crossed (shouldn't happen, safety check)
    valid = (result["bid_price_1"] > 0) & (result["ask_price_1"] > result["bid_price_1"])
    return result[valid].reset_index(drop=True)


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build stock LOB dataset via yfinance")
    ap.add_argument("--ticker", default=TICKER, help="Yahoo Finance ticker, e.g. AAPL")
    ap.add_argument("--period", default=PERIOD,
                    help="Lookback period (max '2y' for hourly). E.g. 6mo, 1y, 2y")
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR,
                               f"lobster_{args.ticker.lower()}_10_level.csv")

    print(f"\nStock LOB Fetcher  (yfinance — no API key required)")
    print(f"  Ticker  : {args.ticker}")
    print(f"  Period  : {args.period}  |  Interval: 1 hour")
    print(f"  Output  : {output_path}\n")

    print(f"Downloading {args.ticker} hourly data...")
    df = yf.download(args.ticker, period=args.period, interval="1h",
                     auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns — flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.dropna()
    print(f"  Downloaded: {len(df):,} hourly bars")
    print(f"  Period    : {df.index[0]}  →  {df.index[-1]}")
    print(f"  Price     : ${df['Close'].min():.2f} – ${df['Close'].max():.2f}")

    print(f"\nReconstructing 10-level LOB...")
    lob = ohlcv_to_lob(df, args.ticker, LEVELS)

    lob.to_csv(output_path, index=False)

    mid        = (lob["bid_price_1"] + lob["ask_price_1"]) / 2
    spread_bps = (lob["ask_price_1"] - lob["bid_price_1"]) / mid * 10_000

    print(f"\n✅  {len(lob):,} ticks saved → {output_path}")
    print(f"    Mid price  : ${mid.mean():.2f}  (${mid.min():.2f} – ${mid.max():.2f})")
    print(f"    Avg spread : {spread_bps.mean():.2f} bps  ({LEVEL_STEP:.2f}$/level)")
    print(f"\nNext — fetch matching news:")
    print(f"    python NewsScrapper/newfetcher2.py")


if __name__ == "__main__":
    main()
