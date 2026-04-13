"""Entry point: `python -m agent.main <command> [options]`.

Thin wrapper that delegates to the Click CLI in `agent.cli`.
"""

from agent.cli import main

if __name__ == "__main__":
    main()
