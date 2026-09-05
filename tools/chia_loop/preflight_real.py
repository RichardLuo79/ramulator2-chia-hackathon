"""Non-billable parity and loaded-candidate isolation checks before real calls."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

from tools.chia_loop import real_eval as E
from tools.chia_loop.real_core import sha
from tools.chia_loop.core import atomic_write_json
from eval import artifacts as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=pathlib.Path)
    args = ap.parse_args()
    root = args.root.resolve()
    tests = E.command([sys.executable, "-m", "pytest",
        "tests/unit_tests/test_chia_real.py", "tests/unit_tests/test_chia_loop.py",
        "tests/unit_tests/test_eval_metrics.py", "tests/unit_tests/test_atomic_skeleton.py", "-q"],
        root / "logs/unit_tests.log")
    parity = []
    for workload in E.TRAIN:
        for model in ("seed", "oracle"):
            direct = root / "parity/simpleo3/DDR5" / workload / model
            protected = root / "training/simpleo3/DDR5" / workload / model
            original = json.loads((direct / "manifest.json").read_text())
            isolated = json.loads((protected / "manifest.json").read_text())
            assert original["per_core_cycles"] == isolated["per_core_cycles"]
            assert original["frontend_stats"] == isolated["frontend_stats"]
            assert original["controller_stats"].keys() == isolated["controller_stats"].keys()
            for key, value in original["controller_stats"].items():
                other = isolated["controller_stats"][key]
                if isinstance(value, float):
                    # print_stats uses six significant digits; Python collect
                    # uses ConfigNode scalar conversion. This is formatting,
                    # not simulation or scored-metric approximation.
                    assert math.isclose(value, other, rel_tol=1e-5, abs_tol=1e-6)
                else:
                    assert value == other
            for file in ("trace.csv.ch0", "controller_trace.csv.ch0"):
                assert A.raw_provenance(direct / file)["sha256"] == A.raw_provenance(protected / file)["sha256"]
            parity.append({"workload": workload, "model": model,
                           "cycles": isolated["per_core_cycles"], "integer_stats_and_trace_bytes_identical": True,
                           "floating_stat_print_tolerance": {"relative": 1e-5, "absolute": 1e-6}})
    probe_dir = root / "security_probe"
    probe_dir.mkdir(exist_ok=True)
    sentinel = root / "private_sentinel.txt"
    sentinel.write_text("private preflight sentinel, not model input\n")
    source = probe_dir / "probe.cpp"
    source.write_text('''#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <cstdio>
#include <cstdlib>
__attribute__((constructor)) void probe() {
  const char* denied[] = {TRACE_PATH, SENTINEL_PATH, "/proc/self/maps"};
  for (auto path : denied) {
    if (open(path, O_RDONLY) >= 0 || open(path, O_WRONLY) >= 0) std::abort();
  }
  if (socket(AF_INET, SOCK_STREAM, 0) >= 0 || fork() >= 0) std::abort();
  char* const args[] = {const_cast<char*>("/usr/bin/true"), nullptr};
  char* const env[] = {nullptr};
  execve(args[0], args, env);
  std::fputs("ISOLATION_PROBE_OK\\n", stderr);
}
'''.replace("TRACE_PATH", json.dumps(E.C.trace_path(E.TRAIN[0])))
        .replace("SENTINEL_PATH", json.dumps(str(sentinel))))
    E.sandbox_command(["/usr/bin/g++", "-O3", "-DNDEBUG", "-fPIC", "-shared", source,
                       "-o", probe_dir / "probe.so"], read=[probe_dir], write=[probe_dir],
                      cwd=probe_dir, log=probe_dir / "compile.log")
    E.run_one(root, E.TRAIN[0], "fixedlat", "loaded_probe", str(probe_dir / "probe.so"), split="preflight")
    log = root / "preflight/simpleo3/DDR5" / E.TRAIN[0] / "loaded_probe/simulation.log"
    assert "ISOLATION_PROBE_OK" in log.read_text()
    assert sentinel.read_text() == "private preflight sentinel, not model input\n"
    E.archive_completed_run(root / "preflight/simpleo3/DDR5" / E.TRAIN[0] / "loaded_probe")
    # Exercise the actual scoped parameter API in a loaded candidate DSO.
    # This fixed-delay interface probe is never exposed as an evolved design.
    from tools.chia_loop import real_core as P
    seed = (root / "seed/atomic_controller.cpp").read_text()
    parameter_probe = seed.replace("void init_model() {}",
        'double probe_delay; void init_model() { probe_delay = model_param("delay", 3, 1, 20); }')
    parameter_probe = parameter_probe.replace("return m_clk + m_latency;", "return m_clk + static_cast<Clk_t>(probe_delay);")
    P.validate_source(seed, parameter_probe)
    parameter_plugin = E.compile_candidate(root, parameter_probe, root / "parameter_probe")
    parameter_run = E.run_one(root, E.TRAIN[0], "candidate", "parameter_probe", parameter_plugin,
        split="preflight", candidate_overrides={"model_parameters": ["delay=7"]})
    assert parameter_run["controller_stats"]["avg_read_latency"] == 7
    E.archive_completed_run(root / "preflight/simpleo3/DDR5" / E.TRAIN[0] / "parameter_probe")
    inputs = {workload: E.C.file_provenance(E.C.trace_path(workload)) for workload in E.TRAIN + E.TEST}
    hashes = [p["sha256"] for p in inputs.values()]
    assert len(set(hashes)) == len(hashes), "trace aliases cross the split"
    atomic_write_json(root / "preflight_pass.json", {
        "passed_at": time.time(), "parity": parity, "loaded_candidate_file_and_network_denial": True,
        "loaded_model_parameter_override": {"default": 3, "override": 7, "measured_read_latency": 7},
        "trace_inputs": inputs, "family_disjoint_split": True,
        "unit_test_command": "pytest tests/unit_tests/test_chia_real.py tests/unit_tests/test_chia_loop.py tests/unit_tests/test_eval_metrics.py tests/unit_tests/test_atomic_skeleton.py -q",
        "unit_test_execution": tests,
        "operator_setup_notes": ["Full-window O3 parity and loaded-candidate isolation checked before paid calls.",
                                 "Live generation preflight is the first counted proposal per arm."]})
    print("PREFLIGHT_PASS: exact parity, loaded-candidate isolation, disjoint trace hashes")


if __name__ == "__main__":
    main()
