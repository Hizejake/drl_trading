"""
Macro Swarm Batch Runner — Run the LLM swarm on GDELT news events and cache vectors.

Reads filtered GDELT events, runs each through the 5-persona LLM swarm,
computes consensus, generates semantic embeddings, and saves everything
to a .npz file for use by the RL environment.

Usage:
    python macro/run_swarm_batch.py                         # Default: 20 events
    python macro/run_swarm_batch.py --max-events 50         # Process 50 events
    python macro/run_swarm_batch.py --input data/raw/filtered_gkg_sample.csv
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
DEFAULT_INPUT = os.path.join(BASE_DIR, "data", "raw", "filtered_gkg_sample.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "data", "raw", "macro_vectors.npz")


def extract_event_texts(csv_path, max_events=20):
    """
    Extract readable event text snippets from GDELT GKG CSV.
    
    GDELT GKG V2 has tab-separated fields. The key fields we use:
    - Field 3 (index 2): Source common name
    - Field 4 (index 3): Document identifier (URL)
    - Column containing themes/persons/organizations
    
    Since GKG doesn't have a clean 'headline' field, we extract
    themes and entity mentions as proxy text for the swarm.
    """
    print(f"[BATCH] Reading events from {os.path.basename(csv_path)}...")
    
    events = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_events * 3:  # Read extra in case we filter some
                break
            
            parts = line.strip().split("\t")
            if len(parts) < 10:
                continue
            
            # GKG V2 fields:
            # 0: GKGRECORDID, 1: V21DATE, 2: V2SourceCollectionIdentifier
            # 3: V2SourceCommonName, 4: V2DocumentIdentifier
            # 7: V2Themes, 8: V2Locations, 9: V2Persons, 10: V2Organizations
            
            source = parts[3] if len(parts) > 3 else "Unknown"
            themes = parts[7] if len(parts) > 7 else ""
            persons = parts[9] if len(parts) > 9 else ""
            orgs = parts[10] if len(parts) > 10 else ""
            
            # Build a readable event text
            theme_list = [t.split(",")[0].replace("_", " ") for t in themes.split(";") if t][:5]
            person_list = [p.split(",")[0] for p in persons.split(";") if p][:3]
            org_list = [o.split(",")[0] for o in orgs.split(";") if o][:3]
            
            event_text = f"Source: {source}. "
            if theme_list:
                event_text += f"Topics: {', '.join(theme_list)}. "
            if person_list:
                event_text += f"People: {', '.join(person_list)}. "
            if org_list:
                event_text += f"Organizations: {', '.join(org_list)}. "
            
            if len(event_text) > 50:  # Skip very short/empty events
                events.append(event_text)
            
            if len(events) >= max_events:
                break
    
    print(f"[BATCH] Extracted {len(events)} event texts")
    return events


async def run_batch(events, delay_between=2.0):
    """Run the swarm on a batch of events with rate limiting."""
    all_results = []
    
    for i, event_text in enumerate(events):
        print(f"\n[BATCH] Event {i+1}/{len(events)}")
        print(f"  Text: {event_text[:100]}...")
        
        try:
            swarm_results = await run_swarm(event_text)
            consensus = aggregate_consensus(swarm_results)
            
            # Count successes
            successes = sum(1 for r in swarm_results if r["error"] is None)
            print(f"  Consensus: dir={consensus['consensus_direction']:.3f}, "
                  f"mag={consensus['avg_magnitude']:.3f}, "
                  f"conf={consensus['avg_confidence']:.3f} "
                  f"({successes}/5 agents succeeded)")
            
            all_results.append({
                "event_text": event_text,
                "consensus": consensus,
                "raw_results": swarm_results,
            })
        except Exception as e:
            print(f"  [ERROR] {e}")
            all_results.append({
                "event_text": event_text,
                "consensus": None,
                "raw_results": None,
            })
        
        # Rate limiting between events
        if i < len(events) - 1:
            print(f"  [WAIT] Sleeping {delay_between}s to avoid rate limits...")
            await asyncio.sleep(delay_between)
    
    return all_results


def save_vectors(results, encoder, output_path):
    """Convert results to numpy arrays and save as .npz."""
    print(f"\n[BATCH] Generating semantic embeddings...")
    
    embeddings = []
    consensus_scalars = []
    event_texts = []
    
    for r in results:
        if r["consensus"] is None:
            continue
        
        c = r["consensus"]
        
        # Generate 384D semantic embedding of combined reasoning
        embedding = encoder.encode(c["combined_reasoning"])
        embeddings.append(embedding)
        
        # Store consensus scalars
        consensus_scalars.append([
            c["consensus_direction"],
            c["avg_magnitude"],
            c["avg_confidence"],
            c["agreement_score"],
        ])
        
        event_texts.append(r["event_text"])
    
    if not embeddings:
        print("[BATCH] No valid results to save!")
        return
    
    embeddings = np.array(embeddings, dtype=np.float32)
    consensus_scalars = np.array(consensus_scalars, dtype=np.float32)
    
    np.savez(
        output_path,
        embeddings=embeddings,
        consensus_scalars=consensus_scalars,
        event_texts=np.array(event_texts, dtype=object),
    )
    
    print(f"[BATCH] Saved to {output_path}")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Consensus scalars: {consensus_scalars.shape}")
    print(f"  Events processed: {len(event_texts)}")


async def main():
    parser = argparse.ArgumentParser(description="Run LLM swarm on GDELT events")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to filtered GKG CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .npz path")
    parser.add_argument("--max-events", type=int, default=20, help="Max events to process")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between events")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        print("Run data/download_gdelt.py first.")
        return
    
    # Extract event texts from GDELT
    events = extract_event_texts(args.input, max_events=args.max_events)
    
    if not events:
        print("[ERROR] No events extracted from input file.")
        return
    
    # Run swarm on all events
    results = await run_batch(events, delay_between=args.delay)
    
    # Generate embeddings and save
    encoder = SemanticEncoder()
    save_vectors(results, encoder, args.output)
    
    print("\n[BATCH] Done! Macro vectors cached for training.")


if __name__ == "__main__":
    asyncio.run(main())
