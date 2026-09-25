"""Vivado-HLS kernel-iteration e2e: the fitness signal on the C++ flow.

Mirror of the RTL-flow loop test, with the expected verdicts inverted:
the vivado_hls skeleton's generated compute() is already a copy and the
generated bare-metal gold model is the identity, so the fresh skeleton
should PASS validation; a deliberately broken kernel should then FAIL it.

    stage A  accgen (vivado flow, size register default 64)
             -> hls (vivado_hls synthesizes dma32/dma64 impl points)
             -> tile config (impl dma64_w32) -> soft + baremetal -> sim
             EXPECT: "... PASS"
    stage B  put_file(espacc.cc with compute writing zeros) -> hls -> sim
             EXPECT: simulation completes but validation FAILS

Stage B's put_file -> hls -> sim sequence is one iteration of the agentic
loop on the C++ flow; its wall-clock time is that loop's feedback latency.

Usage (host, chia env active, cluster up via esp_xcelium.yaml, and
CHIA_ESP_VIVADO_HLS_BIN set in the site env so vivado_hls is on PATH):
    cd <repo root>
    python chia/esp/test/cluster/hls_loop_e2e_test.py

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

COPY_LINE = "_outbuff[i] = _inbuff[i];"
BROKEN_LINE = "_outbuff[i] = 0;"


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
    spec = EspAccelSpec(name=ACC_NAME, flow="vivado",
                        answers_tail=["", "64", "1024"] + [""] * 13)
    kernel_rel = f"{spec.acc_dir}/hw/src/espacc.cc"

    ok = True
    with EspWorkspaceNode(
            reserve_bundle={"CPU": 1, "esp": 1, "esp_xcelium": 1}) as ws:

        print("\n=== stage A: ladder with the generated skeleton (size=64) ===")
        t0 = time.time()
        gen = get(ws.accgen.chia_remote(ESP_ROOT, spec, overwrite=True))
        t_hls = time.time()
        hls = get(ws.accel.chia_remote(ESP_ROOT, ESP_BOARD, spec.make_name, "hls"))
        print(f"hls synthesis took {time.time() - t_hls:.0f}s")
        col = get(ws.collect.chia_remote(ESP_ROOT, [DEFCONFIG]))
        cfg = get(ws.configure.chia_remote(
            ESP_ROOT, ESP_BOARD,
            esp_config=with_acc_tile(col.files[DEFCONFIG], spec.make_name,
                                     ACC_ROW, ACC_COL, impl="dma64_w32")))
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
                [f"soft-build/**/{spec.make_name}*.exe"], max_bytes_per_file=1))
            exes = sorted(exe_col.skipped)
            ok = bool(exes)
            if not ok:
                print(f"no {spec.make_name}*.exe found under soft-build")

        if ok:
            sim_a = get(ws.sim.chia_remote(
                ESP_ROOT, ESP_BOARD, test_program=f"./{exes[0]}",
                pass_pattern="Program Completed", clean=True,
                timeout_seconds=SIM_TIMEOUT))
            verdict_a = _sim_verdict(sim_a)
            print(f"stage A sim: {verdict_a}  ({time.time() - t0:.0f}s total)")
            _tail("stage A transcript", sim_a.stdout, n=1200)
            if verdict_a != "validation-pass":
                print("expected the generated copy skeleton to PASS validation")
                ok = False

        if ok:
            print("\n=== stage B: one loop iteration (broken kernel) ===")
            t1 = time.time()
            src = get(ws.collect.chia_remote(ESP_ROOT, [kernel_rel]))
            kernel = src.files[kernel_rel]
            if COPY_LINE not in kernel:
                print(f"copy line not found in {kernel_rel}; cannot break it")
                ok = False
            else:
                get(ws.put_file.chia_remote(
                    ESP_ROOT, kernel_rel,
                    kernel.replace(COPY_LINE, BROKEN_LINE)))
                hls2 = get(ws.accel.chia_remote(ESP_ROOT, ESP_BOARD,
                                                spec.make_name, "hls"))
                print(f"re-hls   : success={hls2.success}")
                sim_b = get(ws.sim.chia_remote(
                    ESP_ROOT, ESP_BOARD, test_program=f"./{exes[0]}",
                    pass_pattern="Program Completed", clean=True,
                    timeout_seconds=SIM_TIMEOUT))
                verdict_b = _sim_verdict(sim_b)
                print(f"stage B sim: {verdict_b}  (iteration took "
                      f"{time.time() - t1:.0f}s)")
                _tail("stage B transcript", sim_b.stdout, n=1600)
                if not (hls2.success and verdict_b == "validation-fail"):
                    _tail("stage B stderr", sim_b.stderr)
                    ok = False

    print("\n" + ("PASS: HLS copy skeleton passed validation, broken kernel "
                  "failed it — the C++ loop's fitness signal works in both "
                  "directions." if ok else "FAIL: see the markers above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
