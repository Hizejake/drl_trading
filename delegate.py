"""Convenience wrapper so the agent CLI can be run from the repo root.

Usage:
    python delegate.py --provider <hf|openrouter|nvidia> --model <name> --task "..."
"""

from agent.delegate import main


if __name__ == "__main__":
    raise SystemExit(main())