"""Frozen placement qualification, extracted without plotting/cloud operations."""
import json
from . import config
from ramulator_chia.framework.identity import canonical_json, file_sha256
from ramulator_chia.framework.snapshots import publish_bytes

def read(path):
    return json.loads(path.read_text())


def qualify(root):
    """Compare simulated fields, never host-specific paths or elapsed times."""
    protocol = read(root / "protocol.json")
    if read(root / "placement-unit.json").get("passed") is not True:
        raise ValueError("placement unit checks did not pass")
    if protocol.get("completion_policy") == "background-replay":
        replay = read(root / "replay-fixtures/passed.json")
        if set(replay) != {"1", "4", "8"} or any(r["physical_mismatches"] for r in replay.values()):
            raise ValueError("missing continuous-contention qualification")
        watchdog = read(root / "watchdog-fixtures/passed.json")
        if (set(watchdog) != {"positive", "no_progress"} or
                any(r.get("passed") is not True for r in watchdog.values()) or
                watchdog["positive"]["returncode"] != 0 or watchdog["no_progress"]["returncode"] != -6):
            raise ValueError("retirement-progress diagnostics did not pass")
    checks = {}
    for name in protocol["pilots"]:
        for arm in protocol["arms"]:
            receipt = read(root / "local/completed" / name / (arm + ".json"))
            observed = receipt["observation"]
            cores = receipt["identity"]["case"]["num_cores"]
            counts = observed["frontend_stats"]["per_core_instructions"]
            if not observed["complete"] or len(counts) != cores or any(not 20_000_000 <= n < 20_000_005 for n in counts):
                raise ValueError("incomplete qualification: " + name + "/" + arm)
            if arm != "oracle":
                score = read(root / "local/scores" / arm / (name + ".json"))
                if score["pairing_error"] or not score["request"] or score["request_pairing"]["address_mismatch_pairs"] != 0:
                    raise ValueError("request qualification failed: " + name + "/" + arm)
        for arm in ("oracle", "fixedlat"):
            sides = [read(root / site / "completed" / name / (arm + ".json")) for site in ("local", "cloud")]
            signatures = []
            for receipt in sides:
                observed = receipt["observation"]
                trace = observed["traces"]["controller.csv.ch0"]
                signatures.append(dict(case=receipt["identity"]["case"], frontend=observed["frontend_stats"],
                                       controller=observed["controller_stats"],
                                       trace_sha256=trace["logical_sha256"], trace_bytes=trace["logical_bytes"]))
            if signatures[0] != signatures[1]:
                raise ValueError("cross-host simulated results differ: " + name + "/" + arm)
            checks[name + "/" + arm] = signatures[0]
    for cores in (4, 8):
        fixture = read(root / f"fixtures{cores}/passed.json")
        if set(fixture) != {"streaming", "random", "mixed_rw", "page_crossing"}:
            raise ValueError("missing deterministic traffic qualification")
        if any(row.get("physical_mismatches") != 0 or row.get("matched", 0) <= 0 for row in fixture.values()):
            raise ValueError("deterministic traffic qualification failed")
    negative = read(root / "fixtures8/negative-checks.json")
    if set(negative) != {"alias", "out_of_range", "missing_root", "capacity"}:
        raise ValueError("missing fail-closed placement checks")
    if any(row.get("returncode") in (None, 0) for row in negative.values()):
        raise ValueError("invalid placement was not rejected")
    result = dict(protocol_sha256=file_sha256(root / "protocol.json"), passed=True,
                  checks=checks, cross_host_equal="per-core instructions/cycles and complete controller CSV bytes",
                  local_all_eight_arms=True, physical_address_mismatches=0)
    publish_bytes(root / "qualification-passed.json", canonical_json(result).encode())
    return result


