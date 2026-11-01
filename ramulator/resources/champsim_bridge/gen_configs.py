#!/usr/bin/env python3
"""Generate expanded Ramulator configs for the ChampSim bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ramulator

STANDARDS = {
    "DDR5": ("DDR5_16Gb_x8", "DDR5_4800AN", {}),
    "LPDDR5": ("LPDDR5_16Gb_x16", "LPDDR5_6400", {"channel_width": 32}),
    "LPDDR6": ("LPDDR6_16Gb_x12", "LPDDR6_10667_BL24", {}),
    "HBM4": ("HBM4_32Gb_8Hi", "HBM4_8000Mbps", {"channel_width": 64}),
}

def make_controllers(standard: str):
    organization, timing, extra = STANDARDS[standard]

    def dram():
        return getattr(ramulator.dram, standard)(
            org_preset=organization, timing_preset=timing, **extra
        )

    oracle = ramulator.controller.GenericDDR(
        dram=dram(),
        scheduler=ramulator.scheduler.FRFCFSRowHit(),
        refresh_manager=ramulator.refresh_manager.NoRefresh(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
    )
    candidate = ramulator.controller.Atomic(
        dram=dram(),
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
        refresh="none",
    )
    return oracle, candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--standards",
        nargs="+",
        choices=tuple(STANDARDS),
        default=tuple(STANDARDS),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for standard in args.standards:
        oracle, candidate = make_controllers(standard)
        for label, controller in (("oracle", oracle), ("candidate", candidate)):
            memory = ramulator.memory_system.GenericDRAM(
                clock_ratio=1,
                controllers=[controller],
                channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
            )
            config = {
                "frontend": {"impl": "External", "clock_ratio": 1},
                "memory_system": memory.to_config(),
            }
            output = args.output_dir / f"{label}_{standard}.json"
            if output.exists() and not args.force:
                raise FileExistsError(f"refusing to overwrite {output}; pass --force")
            output.write_text(json.dumps(config, indent=1))
            print(output)


if __name__ == "__main__":
    main()
