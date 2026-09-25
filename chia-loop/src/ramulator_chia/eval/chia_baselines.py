"""Operator-only baseline inventory, execution and cache import.

This uses the campaign's native measurement task and matcher. It does not load
an agent, select a candidate, or change a campaign database. Only the five
established baseline models are accepted. Run from the frozen evaluator tree.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from ramulator_chia.framework.identity import canonical_json, digest_json, file_sha256
from ramulator_chia.framework.snapshots import publish_bytes
from ramulator_chia.recovery import OperationalPause, exclusive_lock

MODELS = ("oracle", "fixedlat", "md1", "wmg1", "mess")
OBSERVATIONS = ("controller.csv.ch0",)


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    publish_bytes(Path(path), canonical_json(value).encode())


def status(path, value):
    """Mutable operator status, never a successful measurement pointer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex)
    temporary.write_text(canonical_json(value) + "\n")
    temporary.replace(path)


def scientific_host(record):
    """Remove only the previously audited trace-format build cosmetics.

    In particular, this retains simulator and probe executable hashes, source
    hashes, compiler flags, frontend geometry, and all other settings.
    """
    result = copy.deepcopy(record)
    probe = result["settings"]["format_probe"]
    argv = probe["argv"]
    if (len(argv) != 8 or argv[6] != "-o"
            or Path(argv[5]).name != "trace_format.cpp"
            or Path(argv[7]).name != "trace_format"
            or not Path(argv[5]).is_absolute()
            or Path(argv[5]).parent != Path(argv[7]).parent
            or probe["returncode"] != 0):
        raise ValueError("unrecognized trace-format probe")
    log = result["assets"]["trace-format-build.log"]
    if (log["name"] != "trace-format-build.log" or log["codec"] != "none"
            or log["executable"] or log["stored_sha256"] != probe["log_sha256"]
            or log["logical_sha256"] != probe["log_sha256"]
            or log["logical_bytes"] != log["stored_bytes"]
            or not isinstance(probe["wall_seconds"], (int, float))
            or not 0 <= probe["wall_seconds"] < float("inf")):
        raise ValueError("unrecognized probe log or timing")
    argv[5], argv[7] = "<probe>/trace_format.cpp", "<probe>/trace_format"
    del probe["wall_seconds"], probe["log_sha256"]
    # Keep the member's codec, name, and executable flag checked above.
    for key in ("stored_sha256", "logical_sha256", "stored_bytes", "logical_bytes"):
        del log[key]
    return result


def equivalence(source, target, source_host, target_host):
    if source["model"] not in MODELS or source["candidate"] is not None:
        raise ValueError("baseline import cannot accept a candidate")
    a, b = copy.deepcopy(source), copy.deepcopy(target)
    for identity, record in ((a, source_host), (b, target_host)):
        if identity["host"] != {"kind": "champsim", "receipt_sha256": digest_json(record)}:
            raise ValueError("host record is not the one bound by the measurement")
    a.pop("host"); b.pop("host")
    if a != b or scientific_host(source_host) != scientific_host(target_host):
        raise ValueError("unapproved scientific or runtime identity difference")
    return {"policy": "baseline-probe-cosmetics-v1", "source_identity": source,
            "target_identity": target, "source_host": source_host,
            "target_host": target_host,
            "scientific_host_sha256": digest_json(scientific_host(source_host))}


def observed_signature(receipt):
    """Compare simulated behavior, never host timing or gzip timestamps."""
    return {**{key: receipt.get(key) for key in (
        "case_sha256", "runtime_sha256", "model", "limits", "optimization",
        "binary_sha256", "core_metric", "per_core_cycles", "frontend_stats", "controller_stats")},
        "observations": {name: {k: member[k] for k in ("logical_sha256", "logical_bytes")}
                         for name, member in receipt["traces"].items()}}


def verify(root, wrapper):
    from ramulator_chia.framework.measurement_reports import verify_native_measurement

    directory = Path(root) / wrapper["directory"]
    if not directory.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("measurement escaped its evidence root")
    receipt = verify_native_measurement(directory, wrapper["receipt_sha256"],
                                        frontend="champsim", observations=OBSERVATIONS)
    identity = wrapper["identity"]
    if wrapper["observation"] != receipt or any(receipt[k] != v for k, v in (
        ("model", identity["model"]), ("case", identity["case"]),
        ("runtime_sha256", identity["runtime"]), ("host", identity["host"]),
        ("limits", identity["limits"]))):
        raise ValueError("measurement wrapper contradicts its evidence")
    if identity["candidate"] is not None or identity["model"] not in MODELS:
        raise ValueError("only baseline measurements may be reused")
    if receipt.get("candidate") is not None or receipt.get("calibration") != identity["curve"]:
        raise ValueError("candidate/calibration evidence contradicts the baseline identity")
    return receipt


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError("expected immutable regular evidence: " + str(source))
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno != 18:  # EXDEV, not an excuse to hide permission/existence errors.
            raise
        shutil.copy2(source, destination)


def install(source_root, wrapper, target_root, target_identity, source_host, target_host,
            *, execution_provenance=None):
    """Publish under the evaluator's own lock, with result.json strictly last.

    Partial imports remain unpublished attempts. A concurrent evaluator wins
    normally; its settled result is verified and is never overwritten.
    """
    proof = equivalence(wrapper["identity"], target_identity, source_host, target_host)
    original = verify(source_root, wrapper)
    target_root = Path(target_root)
    directory = target_root / "measurements" / digest_json(target_identity)
    with exclusive_lock(directory / "job.lock"):
        pointer = directory / "result.json"
        if pointer.exists():
            existing = read(pointer)
            if existing["identity"] != target_identity:
                raise ValueError("cache pointer has another identity")
            if observed_signature(verify(target_root, existing)) != observed_signature(original):
                raise ValueError("existing completed result disagrees")
            return existing
        # The final path is deterministic across local/cloud campaign copies,
        # so backup does not create three differently named copies of one trace.
        settled = directory / ("reuse-" + digest_json({
            "source": wrapper["receipt_sha256"], "target": target_identity}))
        attempt = directory / (".import-" + uuid4().hex)
        attempt.mkdir()
        source = Path(source_root) / wrapper["directory"]
        evidence = set(original["files"]) | {m["name"] for m in original["traces"].values()}
        for name in sorted(evidence):
            if not (source / name).resolve().is_relative_to(source.resolve()):
                raise ValueError("evidence escaped original receipt")
            link_or_copy(source / name, attempt / name)
        provenance = "reuse/" + wrapper["receipt_sha256"] + "/"
        publish(attempt / (provenance + "source-result.json"), wrapper)
        link_or_copy(source / "measurement.json", attempt / (provenance + "source-measurement.json"))
        publish(attempt / (provenance + "proof.json"), {**proof, "source_root": str(source_root),
                "original_receipt_sha256": wrapper["receipt_sha256"],
                "original_result_sha256": digest_json(wrapper)})
        derived = copy.deepcopy(original)
        derived["host"] = target_identity["host"]
        derived["baseline_reuse"] = {
            "kind": "derived-reuse-not-an-execution", "proof": provenance + "proof.json",
            "actual_execution_host": original.get("baseline_reuse", {}).get(
                "actual_execution_host", original["host"]),
            "execution_machine": original.get("baseline_reuse", {}).get(
                "execution_machine", execution_provenance),
            "original_receipt_sha256": wrapper["receipt_sha256"]}
        for name in (provenance + p for p in ("source-result.json", "source-measurement.json", "proof.json")):
            derived["files"][name] = file_sha256(attempt / name)
        publish(attempt / "measurement.json", derived)
        result = {"identity": target_identity, "directory": str(attempt.relative_to(target_root)),
                  "receipt_sha256": file_sha256(attempt / "measurement.json"),
                  "observation": derived}
        verify(target_root, result)
        result["directory"] = str(settled.relative_to(target_root))
        if settled.exists():
            # An interruption after rename but before pointer publication.
            verify(target_root, result)
        else:
            attempt.rename(settled)
        publish(pointer, result)
        return result


def load_cases(root):
    from ramulator_chia.framework.config import Evaluation
    from ramulator_chia.framework.inputs import prepare
    from ramulator_chia.framework.archive import describe_payload
    from ramulator_chia.framework.evaluation import SimulationLimits
    from ramulator_chia.framework.candidate import runtime_inputs

    settings = read(root / "baseline-settings.json")
    runtime_inputs(Path(settings["runtime"]))
    configuration = SimpleNamespace(experiment=SimpleNamespace(
        evaluation=Evaluation.model_validate(settings["evaluation"])))
    cases, hosts = prepare(configuration, root)
    curve = describe_payload(Path(settings["mess_curve"]), "inputs/mess.txt")
    limits = SimulationLimits(**settings["limits"])
    return settings, {c.workload: c for group in cases.values() for c in group}, hosts, curve, limits


def identity_for(runtime, case, model, host, limits, curve):
    return {"runtime": file_sha256(runtime / "runtime_manifest.json"), "case": case.identity(),
            "model": model, "limits": asdict(limits), "candidate": None,
            "curve": asdict(curve.member) if model == "mess" else None,
            "host": host.identity()}


def freeze(root, campaign_roots):
    """Verify the existing independent executions and freeze the complete matrix."""
    settings, cases, hosts, curve, limits = load_cases(root)
    runtime = Path(settings["runtime"])
    indexes = []
    for campaign in campaign_roots:
        records = {}
        for path in (campaign / "measurements").glob("*/result.json"):
            wrapper = read(path); identity = wrapper["identity"]
            if (identity["model"] in MODELS and identity["candidate"] is None
                    and identity["case"]["frontend"] == "champsim"
                    and identity["case"]["workload"] in cases):
                key = (identity["case"]["workload"], identity["model"])
                if key in records:
                    raise ValueError("ambiguous completed baseline: " + str(key))
                records[key] = wrapper
        indexes.append(records)
    rows = []
    for name, case in cases.items():
        host = hosts[f"champsim:{case.cores}"]
        for model in MODELS:
            identity = identity_for(runtime, case, model, host, limits, curve)
            originals, signature = [], None
            for campaign, records in zip(campaign_roots, indexes):
                wrapper = records.get((name, model))
                if wrapper is None:
                    continue
                source_host = read(campaign / f"frontend-builds/champsim-c{case.cores}/host.json")
                proof = equivalence(wrapper["identity"], identity, source_host, host.record())
                observed = verify(campaign, wrapper)
                current = observed_signature(observed)
                if signature is not None and current != signature:
                    raise ValueError("cross-campaign simulated result disagrees: " + name)
                signature = current
                originals.append({"root": str(campaign), "wrapper": wrapper,
                                  "equivalence": proof})
            if originals:
                first = originals[0]
                install(Path(first["root"]), first["wrapper"], root, identity,
                        first["equivalence"]["source_host"], host.record())
            rows.append({"id": digest_json(identity), "case": name, "model": model,
                         "cohort": case.stage, "cores": case.cores, "identity": identity,
                         "existing": bool(originals),
                         "originals": [{"root": r["root"],
                             "receipt_sha256": r["wrapper"]["receipt_sha256"],
                             "result_sha256": digest_json(r["wrapper"]),
                             "equivalence": r["equivalence"]} for r in originals]})
            print(canonical_json({"inventoried": len(rows), "case": name, "model": model,
                                  "independent_copies": len(originals)}), flush=True)
    if len(rows) != 500 or len(cases) != 100:
        raise ValueError("baseline matrix is not the approved 500 entries")
    manifest = {"schema": 1, "models": MODELS, "measurements": rows,
                "settings_sha256": file_sha256(root / "baseline-settings.json")}
    publish(root / "manifest.json", manifest)
    # Whole cases stay together. Historical runtimes balance estimated work;
    # core count is the fallback where no baseline has run yet.
    existing = {r["case"]: r for r in rows if r["existing"] and r["model"] == "oracle"}
    def weight(case_rows):
        sample = existing.get(case_rows[0]["case"])
        seconds = 900 * case_rows[0]["cores"]
        if sample:
            seconds = read(root / "measurements" / sample["id"] / "result.json")["observation"]["wall_seconds"]
        return seconds * len(case_rows)
    groups = [[r for r in rows if r["case"] == name and not r["existing"]] for name in cases]
    loads, assignments = [0., 0.], [[], []]
    for group in sorted(filter(None, groups), key=weight, reverse=True):
        slot = min(range(2), key=lambda i: loads[i])
        assignments[slot].extend(group); loads[slot] += weight(group)
    for slot, assigned in enumerate(assignments):
        assigned.sort(key=lambda r: (-r["cores"], -weight([r]), r["case"], r["model"]))
        publish(root / f"queue-{slot}.json", {"manifest_sha256": file_sha256(root / "manifest.json"),
                                             "ids": [r["id"] for r in assigned]})


def run_queue(root, slot, workers):
    """Use existing CHIA tasks and failure receipts on one independent Ray node."""
    import ray
    from ramulator_chia.framework.dram import measurement

    if not 1 <= workers <= 120:
        raise ValueError("worker count outside approved envelope")
    settings, cases, hosts, curve, limits = load_cases(root)
    runtime = Path(settings["runtime"])
    manifest = read(root / "manifest.json"); queue = read(root / f"queue-{slot}.json")
    if queue["manifest_sha256"] != file_sha256(root / "manifest.json"):
        raise ValueError("worker queue changed its manifest")
    rows = {r["id"]: r for r in manifest["measurements"]}
    ray.init(num_cpus=workers, include_dashboard=False, _node_ip_address="127.0.0.1",
             object_store_memory=2 * 1024**3, namespace="baseline-precompute")
    pending, active, outcomes = list(queue["ids"]), {}, {}
    qualification = ("training-c1-whisper", "training-c4-05", "training-c8-01")
    revision = manifest.get("baseline_revision")
    qualified_models = manifest["models"] if revision else ["oracle"]
    jobs = [(name, model, root if revision else root / "qualification",
             "qualification:" + name + (":" + model if revision else ""))
            for name in qualification for model in qualified_models]
    qualification_keys = [j[3] for j in jobs]
    jobs += [(rows[k]["case"], rows[k]["model"], root, k) for k in pending]
    pending = list(jobs)
    started = time.time()
    while pending or active:
        while pending and len(active) < workers:
            if revision and not pending[0][3].startswith("qualification:"):
                if not all(k in outcomes for k in qualification_keys):
                    break
                if not read(root / "qualification-result.json")["passed"]:
                    raise RuntimeError("corrected baseline qualification failed; bulk release blocked")
            available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                                 if line.startswith("MemAvailable:"))) * 1024
            if available < 64 * 1024**3 or shutil.disk_usage(root).free < 64 * 1024**3:
                break
            name, model, store, key = pending.pop(0)
            if (root / "job-status" / (key.replace(":", "-") + ".json")).exists():
                old = read(root / "job-status" / (key.replace(":", "-") + ".json"))
                if old["state"] == "failed":
                    outcomes[key] = old; continue  # No blind retry of failed simulations.
            case = cases[name]; host = hosts[f"champsim:{case.cores}"]
            expected = identity_for(runtime, case, model, host, limits, curve)
            if not key.startswith("qualification:") and expected != rows[key]["identity"]:
                raise ValueError("prepared inputs no longer match the manifest")
            ref = measurement.options(num_cpus=1).remote(runtime, store, case, model, limits,
                       None, curve if model == "mess" else None, host)
            active[ref] = (key, name, store)
        done, _ = ray.wait(list(active), timeout=20) if active else ([], [])
        for ref in done:
            key, name, store = active.pop(ref)
            try:
                wrapper = ray.get(ref)
                if key.startswith("qualification:"):
                    if revision:
                        if wrapper["identity"]["model"] == "oracle":
                            original = read(Path(revision["oracle_references"][name]))
                            a, b = observed_signature(wrapper["observation"]), observed_signature(original["observation"])
                            # This is a cross-build qualification, NOT cache reuse.
                            # The manifest records the changed binary/source hashes.
                            for value in (a,b):
                                value.pop("runtime_sha256"); value.pop("binary_sha256")
                            if a != b: raise ValueError("rebuilt oracle changed simulated behavior")
                    else:
                        expected = next(r for r in rows.values() if r["case"] == name and r["model"] == "oracle")
                        original = read(root / "measurements" / expected["id"] / "result.json")
                        if observed_signature(wrapper["observation"]) != observed_signature(original["observation"]):
                            raise ValueError("full-window host qualification differs from the original")
                outcome = {"state": "complete", "receipt_sha256": wrapper["receipt_sha256"]}
            except Exception as exc:
                outcome = {"state": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            outcomes[key] = outcome
            status(root / "job-status" / (key.replace(":", "-") + ".json"), outcome)
        if (all(k in outcomes for k in qualification_keys)
                and not (root / "qualification-result.json").exists()):
            passed = all(outcomes[k]["state"] == "complete" for k in qualification_keys)
            pairing_error = None
            if passed and revision:
                try:
                    for name in qualification:
                        compare_case(root,name,[r for r in rows.values() if r["case"]==name])
                except Exception as exc:
                    passed, pairing_error = False, str(exc)
            publish(root / "qualification-result.json", {
                "passed": passed, "pairing_error": pairing_error,
                "cases": qualification,
                "receipts": {k: outcomes[k] for k in qualification_keys}})
        snapshot = {"time": time.time(), "started": started, "pending": len(pending),
                    "running": len(active), "complete": sum(o["state"] == "complete" for o in outcomes.values()),
                    "failed": sum(o["state"] == "failed" for o in outcomes.values()),
                    "terminal": not pending and not active}
        status(root / "worker-status.json", snapshot)
        print(canonical_json(snapshot), flush=True)
        # The last qualification can finish in this pass. Bulk work is then
        # eligible on the next pass, not evidence of a resource shortage.
        if not active and pending and not done:
            raise RuntimeError("insufficient memory or disk reserve; operator intervention required")
    ray.shutdown()


def compare_case(root, name, rows):
    from ramulator_chia.framework.measurement_reports import compare_transfer

    complete = {r["model"]: read(root / "measurements" / r["id"] / "result.json")
                for r in rows if (root / "measurements" / r["id"] / "result.json").exists()}
    if "oracle" not in complete:
        return []
    results = []
    for model, wrapper in complete.items():
        if model == "oracle":
            continue
        oracle = complete["oracle"]
        path = root / "comparisons" / f"{name}--{model}.json"
        if path.exists():
            result = read(path)
        else:
            score = compare_transfer(root / oracle["directory"], root / wrapper["directory"],
                oracle_receipt_sha256=oracle["receipt_sha256"], model_receipt_sha256=wrapper["receipt_sha256"],
                frontend="champsim", minimum_oracle_owner_reads=10_000, include_tails=True)
            if score.get("request_pairing", {}).get("address_mismatch_pairs") != 0:
                raise ValueError("common logical requests have physical-address mismatches")
            result = {"case": name, "cohort": wrapper["identity"]["case"]["stage"],
                      "cores": wrapper["identity"]["case"]["num_cores"], "model": model,
                      "oracle_receipt_sha256": oracle["receipt_sha256"],
                      "model_receipt_sha256": wrapper["receipt_sha256"], "score": score,
                      "runtime_seconds": wrapper["observation"]["wall_seconds"]}
            publish(path, result)
        results.append(result)
    return results


def compare_all(root, workers):
    if not 1 <= workers <= 8:
        raise ValueError("matching is limited to eight processes")
    rows = read(root / "manifest.json")["measurements"]
    groups = {r["case"]: [] for r in rows}
    for row in rows:
        groups[row["case"]].append(row)
    result, failures = [], {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(compare_case, root, name, group): name for name, group in groups.items()}
        for future in as_completed(pending):
            try:
                result.extend(future.result())
            except Exception as exc:
                failures[pending[future]] = str(exc)
    status(root / "comparison-inventory.json", {"results": result, "failures": failures})


def export_inventory(root):
    """List settled evidence only; transfer this before publishing any imports."""
    files = set()
    for store in (root, root / "qualification"):
        for receipt_path in (store / "measurements").glob("*/*/measurement.json"):
            receipt = read(receipt_path)
            attempt = receipt_path.parent
            files.add(receipt_path)
            files.update(attempt / p for p in receipt.get("files", {}))
            files.update(attempt / p["name"] for p in receipt.get("traces", {}).values())
            pointer = attempt.parent / "result.json"
            if pointer.exists() and read(pointer)["receipt_sha256"] == file_sha256(receipt_path):
                files.add(pointer)
    for name in ("manifest.json", "baseline-settings.json", "queue-0.json", "queue-1.json",
                 "qualification-result.json", "worker-status.json", "comparison-inventory.json"):
        if (root / name).exists(): files.add(root / name)
    files.update((root / "job-status").glob("*.json"))
    files.update((root / "comparisons").glob("*.json"))
    files.update(root.glob("infrastructure-attempt-*/*.json"))
    files.update(root.glob("infrastructure-attempt-*/job-status/*.json"))
    inventory = {}
    for path in sorted(files):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("untrusted export path")
        inventory[str(path.relative_to(root))] = {"bytes": path.stat().st_size,
                                                 "sha256": file_sha256(path)}
    status(root / "export.json", {"time": time.time(), "files": inventory})
    (root / "export-files.txt").write_text("".join(name + "\n" for name in inventory))


def report(root):
    """Nine separate cohort/core-count tables, with missing cases explicit."""
    from ramulator_chia.framework.scoring import aggregate

    rows = read(root / "manifest.json")["measurements"]
    coverage = []
    for row in rows:
        pointer = root / "measurements" / row["id"] / "result.json"
        record = {"id": row["id"], "case": row["case"], "cohort": row["cohort"],
                  "cores": row["cores"], "model": row["model"], "complete": pointer.exists()}
        if pointer.exists():
            result = read(pointer)
            record.update(receipt_sha256=result["receipt_sha256"],
                          directory=result["directory"], wall_seconds=result["observation"]["wall_seconds"],
                          execution_machine=result["observation"].get("baseline_reuse", {}).get("execution_machine"))
        coverage.append(record)
    status(root / "coverage.json", {"required": len(rows), "complete": sum(r["complete"] for r in coverage),
                                    "measurements": coverage})
    comparisons = [read(p) for p in (root / "comparisons").glob("*.json")]
    lines = ["# ChampSim baseline precomputation", "",
             f"Verified inventory: {sum(r['complete'] for r in coverage)}/{len(rows)} measurements.", "",
             "Core errors are averaged equally over cores within a case, then over cases. "
             "Request MAE/L uses the existing eligible oracle-read population, including unpaired reads. "
             "Only shared completed cases are compared. Missing/failed measurements are never zero.", "",
             "| Cohort | Cores | Model | Shared / required cases | Core MAE (%) | Request MAE/L |",
             "|---|---:|---|---:|---:|---:|"]
    aggregates = []
    for cohort in ("training", "validation", "test"):
        for cores in (1, 4, 8):
            required = {r["case"] for r in rows if r["cohort"] == cohort and r["cores"] == cores}
            groups = {model: {r["case"]: r["score"] for r in comparisons
                      if r["cohort"] == cohort and r["cores"] == cores and r["model"] == model}
                      for model in read(root / "manifest.json")["models"] if model != "oracle"}
            shared = required.intersection(*(set(v) for v in groups.values()))
            for model, values in groups.items():
                if shared:
                    summary = aggregate({n: values[n] for n in sorted(shared)},
                        expected_workloads=sorted(shared), stage=cohort,
                        request_objective="champsim_foreground_stable_pairs")
                    aggregates.append({"cohort": cohort, "cores": cores, "model": model, **summary})
                    a = summary["aggregate"]; core = f"{a['cycle_macro_mae_pct']:.4f}"
                    request = a["request_macro_mae_over_L"]
                    request = "unavailable" if request is None else f"{request:.5f}"
                else:
                    core = request = "pending"
                lines.append(f"| {cohort} | {cores} | {model} | {len(shared)} / {len(required)} | {core} | {request} |")
    status(root / "baseline-accuracy.json", {"aggregates": aggregates, "per_case": comparisons})
    (root / "baseline-accuracy.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "compare", "finish", "install", "export", "report"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--campaign", type=Path, action="append", default=[])
    parser.add_argument("--slot", type=int, choices=(0, 1))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); root = args.root.absolute()
    if args.action == "freeze":
        freeze(root, args.campaign)
    elif args.action == "run":
        run_queue(root, args.slot, args.workers)
    elif args.action == "compare":
        compare_all(root, args.workers)
    elif args.action == "finish":
        # Run on the VM: loss of the operator's SSH connection cannot prevent
        # postprocessing. A failed driver is not treated as an endless wait.
        while not (root / "worker-status.json").exists() or not read(root / "worker-status.json")["terminal"]:
            active = subprocess.run(["systemctl", "is-active", "--quiet", "chia-baseline-worker"])
            if active.returncode:
                raise RuntimeError("baseline driver stopped before settling its queue")
            time.sleep(10)
        if not read(root / "qualification-result.json")["passed"]:
            raise RuntimeError("host qualification failed; preserve evidence without publishing")
        compare_all(root, args.workers)
        export_inventory(root)
    elif args.action == "export":
        export_inventory(root)
    elif args.action == "report":
        report(root)
    else:
        from ramulator_chia.framework.external_frontends import ExternalHost

        settings = read(root / "baseline-settings.json")
        if (root / "worker-status.json").exists() and not read(root / "qualification-result.json")["passed"]:
            raise ValueError("worker qualification must pass before cache publication")
        outcomes = []
        for campaign in args.campaign:
            config = read(campaign / "config.json")
            r = config["run"]["resources"]
            limits = {"timeout_seconds": r["simulation_timeout_seconds"],
                      **{k: r[k] for k in ("memory_bytes", "file_bytes", "source_bytes", "gzip_level")}}
            if (config["experiment"]["evaluation"] != settings["evaluation"]
                    or limits != settings["limits"]):
                raise ValueError("target campaign scientific configuration differs")
            target_hosts = {}
            for cores in (1, 4, 8):
                directory = campaign / f"frontend-builds/champsim-c{cores}"
                target_hosts[cores] = ExternalHost(directory, file_sha256(directory / "host.json")).record()
            for row in read(root / "manifest.json")["measurements"]:
                pointer = root / "measurements" / row["id"] / "result.json"
                if not pointer.exists():
                    continue
                source_host = read(root / f"frontend-builds/champsim-c{row['cores']}/host.json")
                target_host = target_hosts[row['cores']]
                identity = {**row["identity"], "host": {"kind": "champsim", "receipt_sha256": digest_json(target_host)}}
                try:
                    receipt = install(root, read(pointer), campaign, identity, source_host, target_host)
                except OperationalPause:
                    outcomes.append({"campaign": str(campaign), "id": row["id"], "pending": "active measurement lock"})
                    continue
                outcomes.append({"campaign": str(campaign), "id": row["id"],
                                 "target_id": digest_json(identity), "receipt_sha256": receipt["receipt_sha256"]})
        status(root / "import-inventory.json", outcomes)


if __name__ == "__main__":
    main()
