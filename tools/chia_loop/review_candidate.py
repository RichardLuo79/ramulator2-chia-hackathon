"""Record a compliance decision without modifying candidate source or scores."""
import argparse
import json
import pathlib
import time

from tools.chia_loop.core import atomic_write_json
from tools.chia_loop.real_core import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=pathlib.Path)
    decision = parser.add_mutually_exclusive_group(required=True)
    decision.add_argument("--approve", action="store_true")
    decision.add_argument("--reject", action="store_true")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewer-kind", choices=("agent", "human"), required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    requested = json.loads((directory / "review_needed.json").read_text())
    digest = sha((directory / "atomic_controller.cpp").read_bytes())
    if digest != requested["source_sha256"]:
        raise RuntimeError("candidate source changed after requesting review")
    if (directory / "review.json").exists():
        raise RuntimeError("review already recorded; decisions cannot be silently replaced")
    atomic_write_json(directory / "review.json", {
        "source_sha256": digest, "approved": args.approve, "reason": args.reason,
        "reviewer": args.reviewer, "reviewer_kind": args.reviewer_kind,
        "time": time.time(), "scope": "compliance only; no modeling repair or tuning advice"})
    print("Approved" if args.approve else "Rejected", digest)


if __name__ == "__main__":
    main()
