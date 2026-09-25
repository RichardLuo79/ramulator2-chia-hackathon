"""Bootstrap repository commands; implementation lives in ramulator_chia."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "chia-loop/src"),
    str(ROOT / "ramulator/python"),
    str(ROOT / "ramulator"),
]

from ramulator_chia.commands import *  # noqa: F403

if __name__ == "__main__":
    main()
