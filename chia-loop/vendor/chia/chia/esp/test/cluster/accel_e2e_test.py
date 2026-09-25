"""End-to-end test of the ESP accelerator ladder on an Xcelium cluster.

Runs the license-free RTL-flow accelerator chain on ONE EspWorkspaceNode
bundle (which must include the ``esp_xcelium`` seat for the final sim):

    ws.accgen            (skeleton for a new RTL accelerator)
      -> ws.accel hls        (package the RTL into the tech library)
      -> ws.collect + with_acc_tile + ws.configure
                             (put the accelerator on the board's empty tile)
      -> ws.build("soft") + ws.accel baremetal
      -> ws.sim TEST_PROGRAM=<acc>.exe

The generated RTL is a non-functional stub (acc_done asserts immediately,
no DMA), so the bare-metal self-test is expected to REPORT validation
errors; the pass criterion here is the flow — the simulation must run the
accelerator test program to "Program Completed". Making the kernel
functionally correct is the agentic loop's job, not this test's.

Usage (host, chia env active, cluster up via esp_xcelium.yaml):
    cd <repo root>
    python chia/esp/test/cluster/accel_e2e_test.py

Environment overrides:
    ESP_ROOT / ESP_BOARD / ESP_CPU    as in the other drivers
    ESP_ACC_NAME                      accelerator name (default chiatest)
    ESP_ACC_TILE                      "row,col" of the tile to replace
                                      (default 1,0 — vc707 defconfig's empty)
    ESP_SIM_TIMEOUT                   seconds for the final make xmsim
"""

import os
import sys

# cluster -> test -> esp -> chia -> <repo root> == 4 levels up.
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


def _tail(label: str, text: str, n: int = 2500) -> None:
    print(f"--- {label} (tail) ---\n{text[-n:] if text else '(empty)'}\n---")


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

    ok = True
    bdir = board_dir(ESP_ROOT, ESP_BOARD)
    print(f"esp root: {ESP_ROOT}\nboard   : {ESP_BOARD}\nacc     : {ACC_NAME} "
          f"@ tile {ACC_ROW},{ACC_COL}")

    spec = EspAccelSpec(name=ACC_NAME)
    with EspWorkspaceNode(
            reserve_bundle={"CPU": 1, "esp": 1, "esp_xcelium": 1}) as ws:
        print("\n=== stage 1: accgen (RTL-flow skeleton) ===")
        gen = get(ws.accgen.chia_remote(ESP_ROOT, spec, overwrite=True))
        print(f"accgen   : success={gen.success} rc={gen.returncode} "
              f"files={len(gen.listing)}")
        if not gen.success:
            _tail("accgen stdout", gen.stdout)
            _tail("accgen stderr", gen.stderr)
            ok = False

        if ok:
            print("\n=== stage 2: package RTL into tech library ===")
            hls = get(ws.accel.chia_remote(ESP_ROOT, ESP_BOARD, spec.make_name,
                                           "hls"))
            print(f"hls      : success={hls.success} rc={hls.returncode}")
            if not hls.success:
                _tail("hls stdout", hls.stdout)
                _tail("hls stderr", hls.stderr)
                ok = False

        if ok:
            print("\n=== stage 3: accelerator tile + socgen ===")
            col = get(ws.collect.chia_remote(ESP_ROOT, [DEFCONFIG]))
            cfg_text = with_acc_tile(col.files[DEFCONFIG], spec.make_name,
                                     ACC_ROW, ACC_COL)
            cfg = get(ws.configure.chia_remote(ESP_ROOT, ESP_BOARD,
                                               esp_config=cfg_text))
            print(f"configure: success={cfg.success} rc={cfg.returncode}")
            if not cfg.success:
                _tail("esp-config stdout", cfg.stdout)
                _tail("esp-config stderr", cfg.stderr)
                ok = False

        if ok:
            print("\n=== stage 4: software (soft + accelerator baremetal) ===")
            soft = get(ws.build.chia_remote(ESP_ROOT, ESP_BOARD, cpu=ESP_CPU))
            bm = get(ws.accel.chia_remote(ESP_ROOT, ESP_BOARD, spec.make_name,
                                          "baremetal"))
            print(f"soft     : success={soft.success}  "
                  f"baremetal: success={bm.success}")
            if not (soft.success and bm.success):
                _tail("soft stderr", soft.stderr)
                _tail("baremetal stdout", bm.stdout)
                _tail("baremetal stderr", bm.stderr)
                ok = False

        if ok:
            # Locate the built test program rather than assume its path: a
            # 1-byte collect cap puts every matching exe in `skipped`.
            col = get(ws.collect.chia_remote(
                bdir, [f"soft-build/**/{ACC_NAME}*.exe"], max_bytes_per_file=1))
            exes = sorted(col.skipped)
            if not exes:
                print(f"no {ACC_NAME}*.exe found under soft-build; "
                      f"listing: {[k for k in col.listing if 'baremetal' in k][:10]}")
                ok = False

        if ok:
            exe = f"./{exes[0]}"
            print(f"\n=== stage 5: xmsim TEST_PROGRAM={exe} ===")
            # clean: the reconfigure regenerated socmap; stale compiled
            # units otherwise fail elaboration with checksum mismatches.
            sim = get(ws.sim.chia_remote(
                ESP_ROOT, ESP_BOARD, test_program=exe,
                pass_pattern="Program Completed", clean=True,
                timeout_seconds=SIM_TIMEOUT))
            print(f"sim      : success={sim.success} rc={sim.returncode} "
                  f"pass_matched={sim.pass_matched}")
            _tail("xmsim stdout", sim.stdout, n=4000)
            if not sim.success:
                _tail("xmsim stderr", sim.stderr)
                ok = False

    print("\n" + ("PASS: accgen -> hls -> tile config -> baremetal -> xmsim "
                  "ran the accelerator test program."
                  if ok else "FAIL: see the markers above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
