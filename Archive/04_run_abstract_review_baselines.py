#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys


def main() -> None:
    target = Path(__file__).resolve().parent / "paper_review" / "04_run_abstract_review_baselines.py"
    sys.path.insert(0, str(target.parent))
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
