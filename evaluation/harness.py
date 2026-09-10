"""
Evaluation harness wrapper — delegates to the main evaluation_harness.py.

This module exists so the harness can be invoked as:
    python -m evaluation.harness --input ... --output ...

It simply imports and calls the main harness from the project root.
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation_harness import main, run_harness  # noqa: E402

if __name__ == "__main__":
    main()
