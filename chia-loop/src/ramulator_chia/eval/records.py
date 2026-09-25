"""Small JSON/checksum helpers shared by offline study summaries."""

import json
from pathlib import Path

from ramulator_chia.input_data import write_json as save

__all__ = ["read", "save"]


def read(path):
    return json.loads(Path(path).read_text())
