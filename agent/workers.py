"""Worker functions that delegate a prompt to a single LLM provider.

Each function takes (prompt, model) and returns the model's text response.
API keys are loaded from a .env file at import time via python-dotenv.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Fixed, provider-specific base URLs for the OpenAI-compatible endpoints.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _require(env_var: str) -> str:
    """Return the value of an env var or raise a clear error if it's missing."""
    value = os.getenv(env_var)
    if not value:
        raise RuntimeError(
            f"{env_var} is not set. Copy .env.example to .env and fill it in."
        )
    return value


def call_huggingface(prompt: str, model: str) -> str:
    """Call a Hugging Face hosted model via huggingface_hub's InferenceClient."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=_require("HF_API_KEY"))
    completion = client.chat_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content or ""


def call_openrouter(prompt: str, model: str) -> str:
    """Call an OpenRouter model via the OpenAI-compatible chat completions API."""
    from openai import OpenAI

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=_require("OPENROUTER_API_KEY"),
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content or ""


def call_nvidia(prompt: str, model: str) -> str:
    """Call an NVIDIA NIM model via the OpenAI-compatible chat completions API."""
    from openai import OpenAI

    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=_require("NVIDIA_API_KEY"),
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content or ""
