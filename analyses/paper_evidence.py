"""Offline, fail-closed loading and reconciliation of the paper figure bundle."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics

from paper_checks import close, require, check_convergence, check_extensions, check_speed_study

SNAPSHOT = "paper-v1"
ARMS = {"fixedlat", "md1", "wmg1", "mess", "astra_single_core", "deepseek_single_core",
        "gemini_single_core", "opus_single_core"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe bundle path")
    path = root / relative
    require(path.is_file() and not path.is_symlink(), "Missing or linked input: " + str(relative))
    require(path.resolve().is_relative_to(root.resolve()), "Input escapes bundle")
    return path


def validate(data):
    base, additions, convergence, extensions = (data[k] for k in
                                               ("evidence", "additions", "convergence", "extensions"))
    rows = base["single_core_test"]
    require({r["arm"] for r in rows} == ARMS, "Incorrect single-core model coverage")
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        require(len(group) == len({r["case"] for r in group}) == 24, "Incomplete single-core population")
        require({r["case"] for r in group} == {r["case"] for r in rows if r["arm"] == "mess"},
                "Different single-core test membership")
        headline = next(r for r in base["tables"]["foundation_headlines"]
                        if r["arm"] == arm and r["cohort"] == "test")
        for metric in ("core_abs_pct", "mae_L", "oracle_coverage"):
            close(statistics.mean(r[metric] for r in group), float(headline[metric]))
        for r in group:
            require(r["address_mismatches"] == 0 and r["L"] > 0 and r["pairs"] > 0,
                    "Invalid request population")
            require(r["cohort"] == "test" and r["cores"] == 1, "Wrong accuracy population")
            close(r["mae_L"], r["mae_cycles"] / r["L"])
            close(r["oracle_coverage"], r["pairs"] / r["oracle_recorded_reads"])
            require(len(r["model_receipt"]) == len(r["oracle_receipt"]) == 64, "Missing receipt binding")
    require(all(p["equal"] for p in base["oracle_equivalence"]), "Oracle equivalence failed")
    for name, source in base["sources"].items():
        require(hashlib.sha256(source["source"].encode()).hexdigest() ==
                source["candidate"]["files"]["model.cpp"]["sha256"], "Model source hash mismatch: " + name)
    check_convergence(convergence, base["sources"], base["tables"]["foundation_headlines"])
    require({c["batch"] for c in additions["speed_cohorts"]} == {"batch_a", "batch_b"},
            "Missing standalone timing batch")
    for cohort in additions["speed_cohorts"]:
        samples = [r for r in additions["speed"] if r["batch"] == cohort["batch"]]
        require({r["arm"] for r in samples} == set(cohort["arms"]), "Incorrect speed model coverage")
        receipts = {f'{r["pattern"]}-r{r["read_percent"]}-i{r["interval"]}--{r["arm"]}--r{p["repeat"]}':
                    p["receipt_sha256"] for r in samples for p in r["repetitions"]}
        check_speed_study({"protocol": {**cohort["protocol"], "models": cohort["arms"], "repetitions": 5},
                           "failed": 0, "pending": 0, "successful": len(cohort["arms"]) * 150,
                           "summary": samples, "provenance": {"models": cohort["models"],
                           "measurement_receipts": receipts}}, base["sources"])
    usage = additions["usage"]
    require(usage["rounds"] == [1, 10], "Usage includes other rounds")
    require({r["campaign"] for r in usage["tables"]["headline"]} == {"astra", "deepseek", "gemini", "opus"},
            "Incomplete usage coverage")
    for row in usage["tables"]["headline"]:
        campaign = row["campaign"]
        require(usage["endpoint_binding"][campaign] ==
                base["sources"][campaign + "_single_core"]["candidate"]["candidate_id"],
                "Usage belongs to another campaign endpoint")
        rounds = [r for r in usage["tables"]["rounds"] if r["campaign"] == campaign]
        require(len(rounds) == 10 and {r["round"] for r in rounds} == set(range(1, 11)), "Missing usage round")
        for metric in ("input_tokens", "output_tokens", "records", "unknown_usage_records",
                       "observed_cost_lower_usd", "observed_cost_upper_usd"):
            for table in ("rounds", "phases"):
                close(sum(r[metric] for r in usage["tables"][table] if r["campaign"] == campaign), row[metric])
        close(row["output_tokens"], row["reasoning_output_tokens"] + row["non_reasoning_output_tokens"])
    check_extensions(extensions)
    for name, source in base["sources"].items():
        require(extensions["sources"][name] == source["candidate"], "Extension endpoint identity changed")


def load_bundle(root, snapshot=SNAPSHOT):
    root = Path(root).resolve()
    bundle = root / "results/paper" / snapshot
    manifest = json.loads(checked_path(bundle, "manifest.json").read_bytes())
    require(manifest["schema"] == "chia-paper-figures-v1" and manifest["snapshot"] == snapshot,
            "Unexpected paper snapshot")
    for relative, entry in manifest["files"].items():
        path = checked_path(bundle, relative)
        require(path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"],
                "Paper evidence checksum mismatch: " + relative)
    for relative, digest in {**manifest["renderers"], **manifest["dependency_files"]}.items():
        require(sha(checked_path(root, relative)) == digest, "Plotting source checksum mismatch: " + relative)
    data = {key: json.loads((bundle / f"paper-{key}.json").read_bytes())
            for key in ("evidence", "additions", "convergence", "extensions")}
    for key in ("additions", "convergence", "extensions"):
        require(data[key]["paper_evidence_sha256"] == sha(bundle / "paper-evidence.json"),
                "Supplement belongs to another paper snapshot")
    require([f["number"] for f in manifest["figures"]] == list(range(1, 9)), "Incomplete figure inventory")
    validate(data)
    return manifest, data


def verify_exports(root, output, manifest):
    """Reconcile every CSV field and preserve the original paper asset hashes."""
    output = Path(output)
    bundle = Path(root) / "results/paper" / manifest["snapshot"]
    records = []
    for figure in manifest["figures"]:
        tables = {}
        for name in figure["reference_tables"]:
            reference = (bundle / "reference-tables" / name).read_text()
            actual = checked_path(output, name).read_text()
            require(list(csv.DictReader(reference.splitlines())) == list(csv.DictReader(actual.splitlines())),
                    "Generated table differs from paper: " + name)
            tables[name] = sha(output / name)
        files = {ext: sha(checked_path(output, figure["name"] + "." + ext)) for ext in ("pdf", "svg", "png")}
        records.append({"number": figure["number"], "name": figure["name"], "tables_verified": tables,
                        "exports": files, "paper_exports": figure["paper_exports"],
                        "png_matches_paper": files["png"] == figure["paper_exports"]["png"]})
    report = {"schema": "chia-paper-reproduction-v1", "snapshot": manifest["snapshot"],
              "bundle_manifest_sha256": sha(bundle / "manifest.json"), "figures": records}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
