"""Inspect or restore a campaign archive without importing any execution backend.

Examples:
    python -m tools.chia_loop.framework.inspect verify campaign-HASH.zip
    python -m tools.chia_loop.framework.inspect manifest campaign-HASH.zip
    python -m tools.chia_loop.framework.inspect extract campaign-HASH.zip --output new-directory

Extraction is an explicit action. This entry point never runs archived programs.
"""

import argparse
import json
from pathlib import Path

from .archive import ReadLimits, extract, verify


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("action", choices=("verify", "manifest", "extract"))
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--sha256", help="expected outer SHA-256 from a trusted publication receipt"
    )
    parser.add_argument("--output", type=Path, help="new destination directory; extraction only")
    parser.add_argument(
        "--maximum-logical-bytes",
        type=int,
        help="operator limit on total expanded inner payload bytes",
    )
    args = parser.parse_args()
    if (args.action == "extract") != (args.output is not None):
        parser.error("--output is required for extraction and is not used by other actions")
    limits = (
        ReadLimits()
        if args.maximum_logical_bytes is None
        else ReadLimits(maximum_logical_bytes=args.maximum_logical_bytes)
    )
    if args.action == "extract":
        checked = extract(args.archive, args.output, expected_sha256=args.sha256, limits=limits)
    else:
        checked = verify(args.archive, expected_sha256=args.sha256, limits=limits)
    if args.action == "manifest":
        print(json.dumps(checked["manifest"], indent=2, sort_keys=True))
    else:
        members = checked["manifest"]["members"]
        print(
            json.dumps(
                {
                    "verified": True,
                    "archive_sha256": checked["archive_sha256"],
                    "members": len(members),
                    "stored_payload_bytes": sum(item["stored_bytes"] for item in members),
                    "logical_payload_bytes": sum(item["logical_bytes"] for item in members),
                    "metadata": checked["manifest"]["metadata"],
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
