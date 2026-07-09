"""CLI to delegate a single task to one LLM provider and print the result.

Usage:
    python agent/delegate.py --provider <hf|openrouter|nvidia> \\
        --model <name> --task "<description>"
"""

import argparse
import sys

try:
    from .workers import call_huggingface, call_nvidia, call_openrouter
except ImportError:
    from workers import call_huggingface, call_nvidia, call_openrouter

PROVIDERS = {
    "hf": call_huggingface,
    "openrouter": call_openrouter,
    "nvidia": call_nvidia,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Delegate a task to one LLM provider and print the response."
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=sorted(PROVIDERS),
        help="Which provider/worker to call.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name as the provider expects it.",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="The task/prompt to send to the model.",
    )
    args = parser.parse_args()

    worker = PROVIDERS[args.provider]
    try:
        result = worker(args.task, args.model)
    except Exception as exc:  # surface provider/auth errors clearly on the CLI
        print(f"Error calling {args.provider} ({args.model}): {exc}", file=sys.stderr)
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
