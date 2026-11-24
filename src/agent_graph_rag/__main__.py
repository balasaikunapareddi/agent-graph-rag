"""`python -m agent_graph_rag` — runs the demo (same as the `demo` command)."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["demo"]))
