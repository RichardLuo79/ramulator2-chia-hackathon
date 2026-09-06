"""Explicit, auditable v8 -> v9 transport-only amendment for paused Gemini runs.

Capture while v8 is still installed. Apply only after the exact two-file patch
below is installed and tested. This tool never launches a model, clears STOP,
changes a design/score/prompt/budget, or upgrades test-exposed searches.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil
import time

from tools.chia_loop import gemini_loop as G, real_core as P, recovery as R
from tools.chia_loop.core import atomic_write_json

AMENDMENT = "bounded_retry499_20260905"
EDITS = {
    "tools/chia_loop/generation.py": [
        ("in (408, 429, 500, 502, 503, 504)", "in (408, 429, 499, 500, 502, 503, 504)"),
        ("    root = pathlib.Path(root)\n", "    root = pathlib.Path(root)\n    check_stop(root)\n")],
    "tools/chia_loop/gemini_loop.py": [
        ('"version": "gemini_individual_run_v8"', '"version": "gemini_individual_run_v9"'),
        ('"transport_retry_policy": "network, timeout and remote-protocol errors;',
         '"transport_retry_policy": "network, timeout, remote-protocol and unexpected HTTP 499 errors;')],
}


def transformed(relative, raw):
    text = raw.decode()
    for before, after in EDITS[relative]:
        if text.count(before) != 1:
            raise RuntimeError("expected single-purpose v8 source patch does not match: " + relative)
        text = text.replace(before, after)
    return text.encode()


def protected_hashes(root):
    files = {root / "state.json", root / "ledger.json"}
    for directory in ("candidates", "checkpoints", "prompts", "training/reports", "training/report_receipts"):
        files.update(p for p in (root / directory).rglob("*") if p.is_file() and
                     p.suffix in {".json", ".cpp", ".so", ".md"})
    return {str(p.relative_to(root)): P.sha(p.read_bytes()) for p in sorted(files)}


def rearm_checkpoint(saved, rows, now):
    """Only release the specifically documented provider-499 block, once."""
    identity = saved["identity"]
    matching = [r for r in rows if r.get("operation_key") == identity["operation_key"]]
    if (not saved.get("blocked") or saved.get("complete_call_id") is not None or not matching
            or matching[-1].get("http_status") != 499 or matching[-1].get("error") != saved["blocked"]
            or matching[-1].get("payload_sha256") != identity["payload_sha256"]):
        raise RuntimeError("checkpoint is not the saved provider-499 failure")
    updated = copy.deepcopy(saved)
    updated.pop("blocked")
    # v8 did not count this non-retryable failure. Include it in the unchanged
    # eight-failure cap. The first recovery window starts when recovery is
    # explicitly enabled; an already active outage window is never reset.
    updated["failures"] = saved["failures"] + 1
    updated.setdefault("outage_started", now)
    updated["retry_at"] = max(updated.get("retry_at", 0), now + 60)
    updated["recovery_amendment"] = {"id": AMENDMENT, "at": now,
        "original_call_id": matching[-1]["id"], "original_failure_at": matching[-1].get("settled_at"),
        "previously_uncounted_failure_included": True}
    return updated


def capture(root):
    if not (root / "STOP").exists():
        raise RuntimeError("explicit maintenance STOP required")
    with R.exclusive_lock(root / ".supervisor.lock"), R.exclusive_lock(root / ".runner.lock"):
        R.assert_training_open(root)
        manifest = R.read_json(root / "run_manifest.json")
        if manifest["policy"]["version"] != "gemini_individual_run_v8":
            raise RuntimeError("capture requires the original v8 run")
        G.verify_pinned_run(root, manifest)
        folder = root / "operational_amendments" / AMENDMENT
        folder.mkdir(parents=True, exist_ok=False)
        files = ["run_manifest.json", "state.json", "ledger.json", "supervisor_state.json", "STOP"]
        files += ["protocol/" + name for name in EDITS]
        files += [str(p.relative_to(root)) for p in (root / "checkpoints").glob("*.json")]
        before = {}
        for name in files:
            source, target = root / name, folder / "before" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            before[name] = P.sha(source.read_bytes())
        record = {"id": AMENDMENT, "run_id": root.name, "status": "captured", "captured_at": time.time(),
            "authorization": "operator requested bounded retries after the iteration-20 HTTP 499 diagnosis",
            "before_files": before, "protected_hashes": protected_hashes(root),
            "human_operational_intervention": True, "human_modeling_hints": False,
            "generation_calls_by_amendment": 0}
        atomic_write_json(folder / "record.json", record)
        return record


def apply(root):
    if not (root / "STOP").exists():
        raise RuntimeError("explicit maintenance STOP required")
    with R.exclusive_lock(root / ".supervisor.lock"), R.exclusive_lock(root / ".runner.lock"):
        R.assert_training_open(root)
        folder = root / "operational_amendments" / AMENDMENT
        record = R.read_json(folder / "record.json")
        if record["run_id"] != root.name:
            raise RuntimeError("amendment owner changed")
        if record["status"] == "applied":
            G.verify_pinned_run(root, R.read_json(root / "run_manifest.json"))
            return record
        for name, digest in record["before_files"].items():
            if (P.sha((root / name).read_bytes()) != digest
                    or P.sha((folder / "before" / name).read_bytes()) != digest):
                raise RuntimeError("captured file changed before amendment: " + name)
        if protected_hashes(root) != record["protected_hashes"]:
            raise RuntimeError("scientific/checkpoint state changed during maintenance")
        old = R.read_json(root / "run_manifest.json")
        policy = G.configured_policy(root)
        expected = {**old["policy"], "version": "gemini_individual_run_v9",
            "transport_retry_policy": old["policy"]["transport_retry_policy"].replace(
                "network, timeout and remote-protocol errors;",
                "network, timeout, remote-protocol and unexpected HTTP 499 errors;")}
        if policy != expected:
            raise RuntimeError("amendment would change more than the retry policy")
        patched = {}
        for name, digest in old["protocol_hashes"].items():
            original = (root / "protocol" / name).read_bytes()
            if P.sha(original) != digest:
                raise RuntimeError("frozen protocol was changed before amendment")
            intended = transformed(name, original) if name in EDITS else original
            if (G.REPO / name).read_bytes() != intended:
                raise RuntimeError("unexpected implementation change: " + name)
            if name in EDITS:
                patched[name] = intended
        rows = R.read_json(root / "ledger.json")["calls"]
        checkpoint_updates = {}
        now = time.time()
        for path in (root / "checkpoints").glob("*.json"):
            saved = R.read_json(path)
            if saved.get("blocked") and any(r.get("operation_key") == saved["identity"]["operation_key"]
                    and r.get("http_status") == 499 for r in rows):
                checkpoint_updates[str(path.relative_to(root))] = rearm_checkpoint(saved, rows, now)
        # All validation precedes writes. STOP stays present throughout the
        # multi-file amendment, including if interrupted; launch remains blocked.
        updated = copy.deepcopy(old)
        updated["policy"] = policy
        for name, raw in patched.items():
            (root / "protocol" / name).write_bytes(raw)
            updated["protocol_hashes"][name] = P.sha(raw)
        for name, payload in checkpoint_updates.items():
            atomic_write_json(root / name, payload)
        record.update(status="applied", applied_at=now, from_version="gemini_individual_run_v8",
            to_version=policy["version"], patched_protocol={name: P.sha(raw) for name, raw in patched.items()},
            rearmed_checkpoints={name: P.sha((root / name).read_bytes()) for name in checkpoint_updates},
            tool_sha256=P.sha(pathlib.Path(__file__).read_bytes()))
        expected_protected = {**record["protected_hashes"], **record["rearmed_checkpoints"]}
        if protected_hashes(root) != expected_protected:
            raise RuntimeError("unexpected scientific-state mutation during amendment")
        atomic_write_json(folder / "record.json", record)
        updated.setdefault("operational_amendments", []).append({"id": AMENDMENT,
            "record": str((folder / "record.json").relative_to(root)),
            "record_sha256": P.sha((folder / "record.json").read_bytes())})
        updated.update(human_intervention=True, human_operational_intervention=True, human_modeling_hints=False)
        atomic_write_json(root / "run_manifest.json", updated)
        supervisor = R.read_json(root / "supervisor_state.json")
        supervisor.update(human_intervention=True, human_operational_intervention=True, human_modeling_hints=False)
        atomic_write_json(root / "supervisor_state.json", supervisor)
        G.verify_pinned_run(root, updated)
        G.event(root, "operational_protocol_amendment", amendment=AMENDMENT, from_version="gemini_individual_run_v8",
                to_version=policy["version"], actor="operator-authorized infrastructure maintenance",
                human_modeling_hints=False, rearmed_checkpoints=list(checkpoint_updates))
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "apply"))
    parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    record = {"capture": capture, "apply": apply}[args.action](args.root.resolve(strict=True))
    print(json.dumps({k: record[k] for k in ("run_id", "status", "rearmed_checkpoints") if k in record}))


if __name__ == "__main__":
    main()
