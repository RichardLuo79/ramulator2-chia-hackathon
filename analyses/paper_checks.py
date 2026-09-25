"""Unchanged semantic checks from the verified paper importer."""
import math
import statistics

def close(actual: float, expected: float) -> None:
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12):
        raise ValueError(f"Evidence mismatch: {actual!r} != {expected!r}")

def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)

def check_speed_study(study: dict, sources: dict) -> None:
    """Check each host cohort separately; never substitute another oracle."""
    protocol = study["protocol"]
    points = {(p, r, i) for p in ("streaming", "random") for r in (100, 75, 50)
              for i in (1, 4, 16, 64, 256)}
    if (protocol["warmup_requests"] != 100_000 or protocol["measured_requests"] != 5_000_000
            or protocol["repetitions"] != 5 or study["failed"] or study["pending"]
            or study["successful"] != len(protocol["models"]) * 150):
        raise ValueError("Incomplete speed protocol")
    oracle = {(r["pattern"], r["read_percent"], r["interval"]): r
              for r in study["summary"] if r["arm"] == "oracle"}
    if set(oracle) != points:
        raise ValueError("Incomplete same-host oracle")
    for arm in protocol["models"]:
        if arm in sources:
            candidate = study["provenance"]["models"][arm]["candidate"]
            if candidate != sources[arm]["candidate"]:
                raise ValueError("Speed measurement belongs to a different endpoint")
        rows = [r for r in study["summary"] if r["arm"] == arm]
        if len(rows) != 30 or {(r["pattern"], r["read_percent"], r["interval"]) for r in rows} != points:
            raise ValueError("Duplicate or missing speed point")
        for r in rows:
            reps = r["repetitions"]
            if r["n"] != 5 or len(reps) != 5 or {v["repeat"] for v in reps} != set(range(1, 6)):
                raise ValueError("Incomplete speed repetitions")
            for rep in reps:
                if rep["measured_reads_completed"] + rep["measured_writes_completed"] != 5_000_000:
                    raise ValueError("Incomplete measured callbacks")
                close(rep["requests_per_second"], 5_000_000 / rep["wall_seconds"])
                key = f'{r["pattern"]}-r{r["read_percent"]}-i{r["interval"]}--{arm}--r{rep["repeat"]}'
                if len(study["provenance"]["measurement_receipts"].get(key, "")) != 64:
                    raise ValueError("Missing speed receipt identity")
            for field in ("simulated_cycles", "measured_simulated_cycles", "measured_reads_completed",
                          "measured_writes_completed", "admission_wait_sum", "drain_simulated_cycles"):
                if len({rep[field] for rep in reps}) != 1:
                    raise ValueError(f"Nondeterministic simulated counter: {field}")
            close(r["median"], statistics.median(v["requests_per_second"] for v in reps))
            close(r["mean"], statistics.mean(v["requests_per_second"] for v in reps))
            o = oracle[r["pattern"], r["read_percent"], r["interval"]]
            close(r["oracle_relative_speedup"], r["median"] / o["median"])

def check_convergence(payload: dict, sources: dict, headlines: list[dict]) -> None:
    """Keep actual selections, including regressions; never take running minima."""
    campaigns = ("astra", "deepseek", "gemini", "opus")
    rows, decisions = payload["rows"], payload["decisions"]
    splits = ("training", "validation")
    expected = {(c, s, n, k) for c in campaigns for s in splits for k, rounds in
                (("retained", range(11)), ("submitted", range(1, 11))) for n in rounds}
    keys = [(r["campaign"], r["split"], r["round"], r["kind"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete or duplicated single-core convergence")
    if any(r["stage"] != "single_core" or r["cores"] != 1 or r["split"] not in splits for r in rows):
        raise ValueError("Convergence includes another stage, core count, or split")
    index = dict(zip(keys, rows))
    dkeys = [(r["campaign"], r["round"]) for r in decisions]
    if len(dkeys) != 40 or set(dkeys) != {(c, n) for c in campaigns for n in range(1, 11)}:
        raise ValueError("Incomplete convergence decisions")
    decision = dict(zip(dkeys, decisions))
    for r in rows:
        if any(not math.isfinite(r[m]) or r[m] <= 0 for m in ("core_abs_pct", "mae_L")):
            raise ValueError("Nonpositive or nonfinite convergence metric")
    for c in campaigns:
        for n in range(1, 11):
            d = decision[c, n]
            for split in splits:
                selected = index[c, split, n, "retained"]
                reference = index[c, split, n, "submitted"] if d["promoted"] else index[c, split, n - 1, "retained"]
                for metric in ("core_abs_pct", "mae_L"):
                    close(selected[metric], reference[metric])
                if selected["promoted"] != d["promoted"] or index[c, split, n, "submitted"]["promoted"] != d["promoted"]:
                    raise ValueError("Convergence promotion flag differs from committed decision")
            expected_id = d["candidate"] if d["promoted"] else (decision[c, n - 1]["selected"] if n > 1 else None)
            if expected_id is not None and d["selected"] != expected_id:
                raise ValueError("Convergence selection differs from committed decision")
        if decision[c, 10]["selected"] != sources[c + "_single_core"]["candidate"]["candidate_id"]:
            raise ValueError("Convergence ends at a different stage endpoint")
        for split in splits:
            headline = next(r for r in headlines if r["arm"] == c + "_single_core" and r["cohort"] == split)
            for metric in ("core_abs_pct", "mae_L"):
                close(index[c, split, 10, "retained"][metric], float(headline[metric]))
                close(index[c, split, 0, "retained"][metric], index["astra", split, 0, "retained"][metric])

def extension_headlines(data: dict) -> dict:
    """Equal case means, with explicit stage/core/ending populations."""
    tables = {}
    keys = {"gem5": ("arm", "mode", "ending"), "hardware": ("arm", "point", "cores"),
            "multicore": ("arm", "cores")}
    for study, fields in keys.items():
        groups = {}
        for row in data[study]:
            require(row.get("accepted", True) and row.get("address_mismatches", 0) == 0,
                    "Incomplete or physically inconsistent comparison")
            group = tuple(row[f] for f in fields)
            groups.setdefault(group, []).append(row)
        table = []
        for key, rows in sorted(groups.items()):
            count = ({("SE", "roi_end"): 6, ("FS", "roi_end"): 4,
                      ("FS", "instruction_cap"): 5}[key[1:]] if study == "gem5"
                     else (24 if study == "hardware" else 8))
            require(len(rows) == len({r["case"] for r in rows}) == count,
                    "Incomplete or duplicated " + study + " population")
            result = dict(zip(fields, key), cases=count)
            for metric in (("core_abs_pct",) if study == "gem5" else ("core_abs_pct", "mae_L")):
                values = [float(r[metric]) for r in rows]
                require(all(math.isfinite(v) and v >= 0 for v in values), "Nonfinite or negative error")
                result[metric] = statistics.mean(values)
            if study != "gem5":
                result.update(oracle_coverage=statistics.mean(r["oracle_coverage"] for r in rows),
                              matched_reads=sum(r["pairs"] for r in rows))
            table.append(result)
        tables[study] = table
    return tables

def check_extensions(data: dict) -> None:
    """Offline publication checks; never infer a missing result or endpoint."""
    base = {"fixedlat", "md1", "wmg1", "mess"}
    expected = {"gem5": base | {"astra_single_core", "deepseek_single_core", "opus_single_core"},
        "hardware": {"astra_single_core", "deepseek_single_core"},
        "multicore": base | {c + "_" + s for c in ("astra", "deepseek", "gemini")
                              for s in ("single_core", "multicore")}}
    for study in expected:
        require({r["arm"] for r in data[study]} == expected[study], "Wrong extension model availability")
        for row in data[study]:
            if row["arm"] not in base:
                candidate = data["sources"][row["arm"]]
                require(row["candidate_id"] == candidate["candidate_id"]
                        and row["source_sha256"] == candidate["files"]["model.cpp"]["sha256"],
                        "Score belongs to a different selected source")
    require(extension_headlines(data) == data["headlines"], "Extension aggregate changed")
    for study in expected:
        for arm in expected[study]:
            rows = [r for r in data[study] if r["arm"] == arm]
            if study == "gem5":
                require({r["case"] for r in rows} == set(data["populations"]["gem5"]), "Different gem5 workload cohort")
                for row in rows:
                    require([row["mode"], row["ending"]] == data["populations"]["gem5"][row["case"]],
                            "Changed gem5 ending or execution mode")
            elif study == "hardware":
                points = {o + "_q" + str(q) for o in ("DDR5_16Gb_x8", "DDR5_8Gb_x8", "DDR5_16Gb_x16")
                          for q in (32, 64, 128)}
                require({r["point"] for r in rows} == points, "Missing hardware organization/queue setting")
                for point in points:
                    require({r["case"] for r in rows if r["point"] == point} == set(data["populations"]["single_core"]),
                            "Different hardware workload cohort")
            else:
                require({r["cores"] for r in rows} == {4, 8}, "Mixed or missing multicore groups")
                for cores in (4, 8):
                    require({r["case"] for r in rows if r["cores"] == cores} == set(data["populations"][str(cores)]),
                            "Different multicore mix membership")
                for row in rows:
                    require(len(row["per_core_signed_pct"]) == row["cores"], "Incorrect per-core population")
                    close(row["core_abs_pct"], statistics.mean(abs(v) for v in row["per_core_signed_pct"]))
                    require(row["L"] > 0 and row["pairs"] > 0 and 0 < row["oracle_coverage"] <= 1,
                            "Invalid request population")
    proof = {(r["arm"], r["case"]): r for r in data["provenance"]["gemini_multicore"]["proofs"]}
    gemini = [r for r in data["multicore"] if r["arm"].startswith("gemini_")]
    require(len(proof) == len(gemini) == 32, "Missing Gemini oracle-equivalence proofs")
    for row in gemini:
        p = proof[row["arm"], row["case"]]
        require(p["oracle_equivalent"] and p["complete_observations_verified"]
                and p["model_receipt"] == row["model_receipt"] and p["oracle_receipt"] == row["oracle_receipt"],
                "Gemini comparison lacks verified measurement binding")
