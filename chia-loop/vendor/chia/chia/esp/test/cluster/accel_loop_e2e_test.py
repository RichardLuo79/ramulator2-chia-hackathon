"""Kernel-iteration e2e: the fitness signal of the agentic accelerator loop.

Demonstrates, through the workspace members an agent would use, that the
accelerator self-test discriminates between a broken and a working kernel:

    stage A  accgen (size register default 64 -> the test moves real data)
             -> hls -> tile config -> soft + baremetal -> sim(clean)
             EXPECT: simulation completes but validation does NOT pass
             (the generated stub moves no data).
    stage B  put_file(copy kernel) -> hls -> sim(clean)
             EXPECT: "... PASS" — the kernel DMA-copies the buffer, so
             validation only succeeds if the hardware actually worked.

Stage B's put_file -> rebuild -> sim sequence is exactly one iteration of
the agentic loop; its wall-clock time is the loop's feedback latency.

Usage (host, chia env active, cluster up via esp_xcelium.yaml):
    cd <repo root>
    python chia/esp/test/cluster/accel_loop_e2e_test.py

Environment overrides: ESP_ROOT / ESP_BOARD / ESP_CPU / ESP_ACC_NAME /
ESP_ACC_TILE / ESP_SIM_TIMEOUT as in accel_e2e_test.
"""

import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 4)))
sys.path.insert(0, _REPO_ROOT)

import ray

from chia.base.ChiaFunction import get
from chia.esp.esp_workspace import EspWorkspaceNode, board_dir, with_acc_tile
from chia.esp.state_def import EspAccelSpec

ESP_ROOT = os.environ.get("ESP_ROOT", "/home/espuser/esp")
ESP_BOARD = os.environ.get("ESP_BOARD", "xilinx-vc707-xc7vx485t")
ESP_CPU = os.environ.get("ESP_CPU", "ariane")
ACC_NAME = os.environ.get("ESP_ACC_NAME", "chiatest")
ACC_ROW, ACC_COL = (int(v) for v in os.environ.get("ESP_ACC_TILE", "1,0").split(","))
SIM_TIMEOUT = int(os.environ.get("ESP_SIM_TIMEOUT", str(7200)))

DEFCONFIG = f"socs/defconfig/esp_{ESP_BOARD}_defconfig"
KERNEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "copy_kernel_basic_dma64.v")


def _tail(label: str, text: str, n: int = 2500) -> None:
    print(f"--- {label} (tail) ---\n{text[-n:] if text else '(empty)'}\n---")


def _sim_verdict(sim) -> str:
    if not sim.pass_matched:
        return "did-not-complete"
    return "validation-pass" if "... PASS" in sim.stdout else "validation-fail"


def main() -> int:
    print(f"[driver] connecting to ray cluster (working_dir={_REPO_ROOT})")
    ray.init(
        address="auto",
        runtime_env={
            "working_dir": _REPO_ROOT,
            "excludes": [".venv/**", ".git/**", "**/__pycache__/**",
                         "**/*.pyc", "**/.pytest_cache/**"],
        },
    )

    # size register: default 64 (real data volume), max 1024.
    spec = EspAccelSpec(name=ACC_NAME,
                        answers_tail=["", "64", "1024"] + [""] * 13)
    kernel = open(KERNEL_FILE).read().replace("chiatest", ACC_NAME)
    kernel_dst = (f"{spec.acc_dir}/hw/src/{spec.make_name}_basic_dma64/"
                  f"{spec.make_name}_basic_dma64.v")

    ok = True
    with EspWorkspaceNode(
            reserve_bundle={"CPU": 1, "esp": 1, "esp_xcelium": 1}) as ws:

        print("\n=== stage A: ladder with the generated stub (size=64) ===")
        t0 = time.time()
        gen = get(ws.accgen.chia_remote(ESP_ROOT, spec, overwrite=True))
        hls = get(ws.accel.chia_remote(ESP_ROOT, ESP_BOARD, spec.make_name, "hls"))
        col = get(ws.collect.chia_remote(ESP_ROOT, [DEFCONFIG]))
        cfg = get(ws.configure.chia_remote(
            ESP_ROOT, ESP_BOARD,
            esp_config=with_acc_tile(col.files[DEFCONFIG], spec.make_name,
                                     ACC_ROW, ACC_COL)))
        soft = get(ws.build.chia_remote(ESP_ROOT, ESP_BOARD, cpu=ESP_CPU))
        bm = get(ws.accel.chia_remote(ESP_ROOT, ESP_BOARD, spec.make_name,
                                      "baremetal"))
        stages = {"accgen": gen, "hls": hls, "configure": cfg, "soft": soft,
                  "baremetal": bm}
        for label, res in stages.items():
            print(f"{label:9s}: success={res.success}")
            if not res.success:
                _tail(f"{label} stdout", res.stdout)
                _tail(f"{label} stderr", res.stderr)
                ok = False

        if ok:
            exe_col = get(ws.collect.chia_remote(
                board_dir(ESP_ROOT, ESP_BOARD),
                [f"soft-build/**/{ACC_NAME}*.exe"], max_bytes_per_file=1))
            exes = sorted(exe_col.skipped)
            ok = bool(exes)
            if not ok:
                print("no accelerator exe found under soft-build")

        if ok:
            sim_a = get(ws.sim.chia_remote(
                ESP_ROOT, ESP_BOARD, test_program=f"./{exes[0]}",
                pass_pattern="Program Completed", clean=True,
                timeout_seconds=SIM_TIMEOUT))
            verdict_a = _sim_verdict(sim_a)
            print(f"stage A sim: {verdict_a}  ({time.time() - t0:.0f}s total)")
            _tail("stage A transcript", sim_a.stdout, n=1200)
            if verdict_a != "validation-fail":
                print("expected the stub to complete but FAIL validation")
                ok = False

        if ok:
            print("\n=== stage B: one loop iteration (copy kernel) ===")
            t1 = time.time()
            get(ws.put_file.chia_remote(ESP_ROOT, kernel_dst, kernel))
            hls2 = get(ws.accel.chia_remote(ESP_ROOT, ESP_BOARD, spec.make_name,
                                            "hls"))
            print(f"reinstall: success={hls2.success}")
            sim_b = get(ws.sim.chia_remote(
                ESP_ROOT, ESP_BOARD, test_program=f"./{exes[0]}",
                pass_pattern="Program Completed", clean=True,
                timeout_seconds=SIM_TIMEOUT))
            verdict_b = _sim_verdict(sim_b)
            print(f"stage B sim: {verdict_b}  (iteration took "
                  f"{time.time() - t1:.0f}s)")
            _tail("stage B transcript", sim_b.stdout, n=1600)
            if not (hls2.success and verdict_b == "validation-pass"):
                _tail("stage B stderr", sim_b.stderr)
                ok = False

    print("\n" + ("PASS: stub failed validation, copy kernel passed it — "
                  "the loop's fitness signal works in both directions."
                  if ok else "FAIL: see the markers above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
