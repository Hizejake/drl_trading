import asyncio
import json
import os
import sys
import numpy as np
from litellm import acompletion
import litellm

# Force UTF-8 output on Windows (avoids cp1252 crashing on Unicode box chars)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── OpenRouter Configuration ──────────────────────────────────────────────────
# LiteLLM routes to OpenRouter via the "openrouter/" prefix.
# The OPENROUTER_API_KEY is read automatically from the .env file by litellm.
OPENROUTER_SITE_URL = "https://github.com/drl-trading"  # for HTTP-Referer header
OPENROUTER_APP_NAME = "DRL-HFT-Swarm"

# ── Persona → Model Assignment ────────────────────────────────────────────────
# Each agent uses a DIFFERENT base model to guarantee genuine swarm heterogeneity.
# All models are confirmed free-tier on OpenRouter as of March 2026.
# LiteLLM's OpenRouter prefix: "openrouter/<provider>/<model>" (no double slash)
PERSONA_MODELS = {
    # Live slugs as of 2026-07 — check GET /api/v1/models (:free) when these drift
    "momentum":       "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "mean_reversion": "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
    "macro_risk":     "openrouter/openai/gpt-oss-120b:free",
    "liquidity":      "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    "volatility":     "openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
}

# LiteLLM reads OPENROUTER_API_KEY automatically when model prefix is 'openrouter/'
# Ensure it is set from .env (load_dotenv already ran above)

# ── Persona System Prompts ────────────────────────────────────────────────────
PERSONAS = {
    "momentum": (
        "You are a momentum trader at a quantitative hedge fund. "
        "Your job is to predict whether the initial price move caused by a news event will CONTINUE over the next 30 minutes. "
        "Respond ONLY with a valid JSON object with exactly these keys: "
        '"direction" (string: "up", "down", or "neutral"), '
        '"magnitude" (float 0.0 to 1.0, how large the expected move is), '
        '"confidence" (float 0.0 to 1.0), '
        '"reasoning" (string: one concise sentence citing your logic).'
    ),
    "mean_reversion": (
        "You are a mean-reversion trader at a quantitative hedge fund. "
        "Your job is to predict whether the initial price move caused by a news event will FADE or REVERSE over the next 30 minutes. "
        "Respond ONLY with a valid JSON object with exactly these keys: "
        '"direction" (string: "up", "down", or "neutral"), '
        '"magnitude" (float 0.0 to 1.0), '
        '"confidence" (float 0.0 to 1.0), '
        '"reasoning" (string: one concise sentence).'
    ),
    "macro_risk": (
        "You are a macro risk analyst at a global investment bank. "
        "Assess the SYSTEMIC RISK and flight-to-quality flows this event could trigger. "
        "Respond ONLY with a valid JSON object with exactly these keys: "
        '"direction" (string: "up", "down", or "neutral" for the affected equity), '
        '"magnitude" (float 0.0 to 1.0), '
        '"confidence" (float 0.0 to 1.0), '
        '"reasoning" (string: one concise sentence).'
    ),
    "liquidity": (
        "You are a microstructure liquidity analyst. "
        "Predict how this news event will affect BID-ASK SPREAD and ORDER BOOK DEPTH over the next 30 minutes. "
        "For 'direction': 'up' means spreads will widen (bad for trading), 'down' means they will tighten. "
        "Respond ONLY with a valid JSON object with exactly these keys: "
        '"direction" (string: "up" or "down" or "neutral"), '
        '"magnitude" (float 0.0 to 1.0), '
        '"confidence" (float 0.0 to 1.0), '
        '"reasoning" (string: one concise sentence).'
    ),
    "volatility": (
        "You are a volatility trader specializing in short-term implied and realized vol. "
        "Predict whether REALIZED VOLATILITY for the affected stock will increase or decrease over the next 30 minutes. "
        "'up' means vol will spike significantly. "
        "Respond ONLY with a valid JSON object with exactly these keys: "
        '"direction" (string: "up", "down", or "neutral"), '
        '"magnitude" (float 0.0 to 1.0), '
        '"confidence" (float 0.0 to 1.0), '
        '"reasoning" (string: one concise sentence).'
    ),
}

# ── Extra headers for OpenRouter analytics ────────────────────────────────────
EXTRA_HEADERS = {
    "HTTP-Referer": OPENROUTER_SITE_URL,
    "X-Title": OPENROUTER_APP_NAME,
}


def _extract_json(text: str) -> dict:
    """Extract the first JSON object from an LLM response (handles code
    fences, reasoning preambles, and trailing prose)."""
    import re
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first balanced {...} block in the text
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


async def _call_agent(persona_name: str, system_prompt: str, event_text: str, model: str):
    """Call a single LLM persona asynchronously via OpenRouter, with retry on rate limits."""
    max_retries = 3
    backoff_seconds = [2, 4, 8]

    for attempt in range(max_retries):
        try:
            response = await acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Analyze this financial news event:\n\n{event_text}"}
                ],
                response_format={"type": "json_object"},
                extra_headers=EXTRA_HEADERS,
                temperature=0.3,
                max_tokens=2048,  # reasoning-style models need headroom before the JSON
            )
            content = response.choices[0].message.content
            if not content:
                return {"persona": persona_name, "model": model, "parsed": None,
                        "error": "empty response content"}
            parsed = _extract_json(content)
            return {"persona": persona_name, "model": model, "parsed": parsed, "error": None}

        except json.JSONDecodeError as e:
            return {"persona": persona_name, "model": model, "parsed": None, "error": f"JSON parse error: {e}"}

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RateLimitError" in err_str or "rate-limited" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = backoff_seconds[attempt]
                print(f"  [RETRY] {persona_name} rate-limited, retrying in {wait}s... (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
                continue
            return {"persona": persona_name, "model": model, "parsed": None, "error": err_str[:120]}


async def run_swarm(event_text: str) -> list[dict]:
    """
    Fire all 5 LLM personas concurrently via asyncio.gather().
    Total latency ≈ slowest single API call, not sum of all calls.
    """
    print(f"\n[SWARM] Firing {len(PERSONAS)} agents concurrently...")
    tasks = [
        _call_agent(name, prompt, event_text, PERSONA_MODELS[name])
        for name, prompt in PERSONAS.items()
    ]
    results = await asyncio.gather(*tasks)
    return results


def aggregate_consensus(swarm_results: list[dict]) -> dict:
    """Compute confidence-weighted consensus vector from swarm responses."""
    dir_map = {"up": 1.0, "down": -1.0, "neutral": 0.0}
    weighted_dir = 0.0
    total_conf = 0.0
    magnitudes, directions, reasoning_parts = [], [], []

    for r in swarm_results:
        if r["error"] is not None:
            print(f"  [WARN] {r['persona']} ({r['model'].split('/')[-1]}) failed: {r['error']}")
            continue
        p = r["parsed"]
        conf = float(p.get("confidence", 0.5))
        d_val = dir_map.get(str(p.get("direction", "neutral")).lower(), 0.0)

        weighted_dir += d_val * conf
        total_conf += conf
        magnitudes.append(float(p.get("magnitude", 0.0)))
        directions.append(d_val)
        reasoning_parts.append(f"[{r['persona']}]: {p.get('reasoning', '')}")

    consensus_direction = (weighted_dir / total_conf) if total_conf > 0 else 0.0
    avg_magnitude   = float(np.mean(magnitudes)) if magnitudes else 0.0
    avg_confidence  = (total_conf / len(swarm_results)) if swarm_results else 0.0
    agreement_score = float(1.0 - np.std(directions)) if len(directions) > 1 else 1.0
    combined_reasoning = " ".join(reasoning_parts)

    return {
        "consensus_direction": round(consensus_direction, 4),
        "avg_magnitude":       round(avg_magnitude, 4),
        "avg_confidence":      round(avg_confidence, 4),
        "agreement_score":     round(agreement_score, 4),
        "combined_reasoning":  combined_reasoning,
    }


class SemanticEncoder:
    """Encodes swarm reasoning text into a 384-dim dense vector using all-MiniLM-L6-v2."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[ENCODER] Loading SentenceTransformer: {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        """Returns a (384,) float32 numpy array."""
        return self.model.encode(text, convert_to_numpy=True)


# ── Live test ─────────────────────────────────────────────────────────────────
async def test_swarm():
    sample_event = (
        "Apple (AAPL) reports Q2 earnings: EPS of $1.52 beats Wall Street estimate of $1.43 "
        "by 6.3%. Revenue of $94.4B slightly misses $94.7B consensus. "
        "iPhone sales down 2% YoY. Services revenue up 14% YoY. Guidance raised."
    )

    results = await run_swarm(sample_event)

    print("\n=== Swarm Responses ===================================================")
    for r in results:
        if r["error"]:
            print(f"  [{r['persona']:15s}] FAILED: {r['error'][:80]}")
        else:
            p = r["parsed"]
            model_short = r["model"].split("/")[-1].replace(":free", "")
            dir_str = str(p.get("direction", "?")).upper()
            print(f"  [{r['persona']:15s}] {dir_str:7s}  Mag={float(p.get('magnitude',0)):.2f}"
                  f"  Conf={float(p.get('confidence',0)):.2f}  ({model_short})")
    print("=======================================================================")

    consensus = aggregate_consensus(results)
    print("\n--- Consensus Summary --------------------------------------------------")
    for k, v in consensus.items():
        if k != "combined_reasoning":
            print(f"  {k:25s}: {v}")

    encoder = SemanticEncoder()
    embedding = encoder.encode(consensus["combined_reasoning"])
    print(f"\n  Semantic embedding shape: {embedding.shape}   (dtype: {embedding.dtype})")
    print("  First 5 dims:", embedding[:5].round(4))


if __name__ == "__main__":
    asyncio.run(test_swarm())
