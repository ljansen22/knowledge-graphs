"""
Entry point — regenerates the dataset.
Run: uv run python main.py
"""

import subprocess
import sys


def main():
    print("Regenerating dataset…")
    result = subprocess.run(
        [sys.executable, "data/generate_dataset.py"],
        check=True,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
