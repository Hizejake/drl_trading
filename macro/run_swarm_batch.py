"""
Macro Swarm Batch Runner — Run the LLM swarm on news events and cache
time-aligned vectors.

Reads a Polygon-style news CSV (published_utc, title, description, ...),
runs each event through the 5-persona LLM swarm, computes consensus,
generates semantic embeddings, and saves everything to a .npz keyed by
event timestamp for time-aligned replay in the RL environment.

Output .npz keys:
    timestamps  (N,)   float64 epoch seconds (event publish time)
    scalars     (N,4)  float32 [direction, magnitude, confidence, agreement]
    embeddings  (N,384) float32 MiniLM embedding of combined reasoning
    event_texts (N,)   object

Usage:
    python macro/run_swarm_batch.py                         # Default: 20 events
    python macro/run_swarm_batch.py --max-events 50         # Process 50 events
    python macro/run_swarm_batch.py --input data/raw/polygon_aapl_news.csv
"""

import asyncio
import os
import sys
import argparse
import numpy as np
import pandas as pd

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from macro.swarm import run_swarm, aggregate_consensus, SemanticEncoder

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, "data", "raw", "polygon_aapl_news.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "data", "raw", "macro_vectors.npz")


def extract_events(csv_path, max_events=20):
    """
    Extract (timestamp, text) events from a Polygon news CSV.

    Events are sampled evenly across the file's full time range so that a
    small --max-events still gives the replay env coverage over the whole
    LOB dataset rather than just the first day.
    """
    print(f"[BATCH] Reading news from {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path)

    required = {"published_utc", "title"}
    missing = required - set(df.columns)
    assert not missing, f"News CSV missing columns: {missing}"

    df = df.dropna(subset=["published_utc", "title"])
    # Unit-independent epoch seconds (pandas may parse as ns or us resolution)
    dt = pd.to_datetime(df["published_utc"], utc=True)
    df["epoch"] = (dt - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
    df = df.sort_values("epoch").reset_index(drop=True)

    if len(df) > max_events:
        idx = np.linspace(0, len(df) - 1, max_events).round().astype(int)
        df = df.iloc[np.unique(idx)]

    events = []
    for _, row in df.iterrows():
        desc = row.get("description", "")
        desc = "" if pd.isna(desc) else str(desc)
        text = f"{row['title']}. {desc}".strip()[:1200]
        events.append({"timestamp": float(row["epoch"]), "text": text})

    print(f"[BATCH] Sampled {len(events)} events spanning "
          f"{pd.to_datetime(events[0]['timestamp'], unit='s')} → "
          f"{pd.to_datetime(events[-1]['timestamp'], unit='s')}")
    return events


async def run_batch(events, delay_between=2.0):
    """Run the swarm on a batch of events with rate limiting."""
    all_results = []

    for i, event in enumerate(events):
        print(f"\n[BATCH] Event {i+1}/{len(events)}")
        print(f"  Text: {event['text'][:100]}...")

        try:
            swarm_results = await run_swarm(event["text"])
            consensus = aggregate_consensus(swarm_results)

            successes = sum(1 for r in swarm_results if r["error"] is None)
            print(f"  Consensus: dir={consensus['consensus_direction']:.3f}, "
                  f"mag={consensus['avg_magnitude']:.3f}, "
                  f"conf={consensus['avg_confidence']:.3f} "
                  f"({successes}/5 agents succeeded)")

            all_results.append({
                "timestamp": event["timestamp"],
                "event_text": event["text"],
                "consensus": consensus if successes > 0 else None,
            })
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_results.append({
                "timestamp": event["timestamp"],
                "event_text": event["text"],
                "consensus": None,
            })

        # Rate limiting between events
        if i < len(events) - 1:
            print(f"  [WAIT] Sleeping {delay_between}s to avoid rate limits...")
            await asyncio.sleep(delay_between)

    return all_results


def save_vectors(results, encoder, output_path):
    """Convert results to numpy arrays and save as .npz."""
    print(f"\n[BATCH] Generating semantic embeddings...")

    timestamps, scalars, embeddings, event_texts = [], [], [], []

    for r in results:
        if r["consensus"] is None:
            continue
        c = r["consensus"]

        embeddings.append(encoder.encode(c["combined_reasoning"]))
        scalars.append([
            c["consensus_direction"],
            c["avg_magnitude"],
            c["avg_confidence"],
            c["agreement_score"],
        ])
        timestamps.append(r["timestamp"])
        event_texts.append(r["event_text"])

    if not embeddings:
        print("[BATCH] No valid results to save!")
        return

    np.savez(
        output_path,
        timestamps=np.array(timestamps, dtype=np.float64),
        scalars=np.array(scalars, dtype=np.float32),
        embeddings=np.array(embeddings, dtype=np.float32),
        event_texts=np.array(event_texts, dtype=object),
    )

    print(f"[BATCH] Saved to {output_path}")
    print(f"  Events: {len(timestamps)}  scalars: {np.array(scalars).shape}  "
          f"embeddings: {np.array(embeddings).shape}")


async def main():
    parser = argparse.ArgumentParser(description="Run LLM swarm on news events")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to Polygon news CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .npz path")
    parser.add_argument("--max-events", type=int, default=20, help="Max events to process")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between events")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        return

    events = extract_events(args.input, max_events=args.max_events)
    if not events:
        print("[ERROR] No events extracted from input file.")
        return

    results = await run_batch(events, delay_between=args.delay)

    encoder = SemanticEncoder()
    save_vectors(results, encoder, args.output)

    print("\n[BATCH] Done! Time-aligned macro vectors cached for training.")


if __name__ == "__main__":
    asyncio.run(main())
