"""End-to-end test for the ESP simulation flow on an Xcelium cluster.

Runs the license-gated chain on ONE EspWorkspaceNode bundle (which must
include the ``esp_xcelium`` seat):

    ws.configure  (board defconfig -> make esp-config)
      -> ws.build("soft")   (bare-metal systest + boot srecs)
      -> ws.sim             (make xmsim: compile RTL + simulate under the
                             batch input script)

The sim transcript tail is printed even on success — the boot messages and
the testbench's termination assert are the human-readable evidence that the
simulated SoC really ran, and matter as much as the exit code.

Usage (host, chia env active, cluster up via esp_xcelium.yaml):
    cd <repo root>
    python chia/esp/test/cluster/xmsim_e2e_test.py

Environment overrides:
    ESP_ROOT / ESP_BOARD / ESP_CPU / ESP_CONFIG_PATH   as in soft_build_e2e_test
    ESP_SIM_TIMEOUT       seconds for make xmsim (default 7200; the first
                          run compiles the Xilinx simlibs + all RTL)
    ESP_SIM_PASS_PATTERN  regex that must appear in the sim transcript for
                          success (default "Program Completed" — the message
                          of the top.vhd assert that ends an ESP simulation)
    ESP_SIM_INPUT         override the batch xmsim.in content (e.g. a
                          bounded "run 10 ms\\nexit\\n")
"""

import os
import sys

# cluster -> test -> esp -> chia -> <repo root> == 4 levels up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 4)))
sys.path.insert(0, _REPO_ROOT)

import ray

from chia.base.ChiaFunction import get
from chia.esp.esp_workspace import EspWorkspaceNode

ESP_ROOT = os.environ.get("ESP_ROOT", "/home/espuser/esp")
ESP_BOARD = os.environ.get("ESP_BOARD", "xilinx-vc707-xc7vx485t")
ESP_CPU = os.environ.get("ESP_CPU", "ariane")
ESP_CONFIG_PATH = os.environ.get(
    "ESP_CONFIG_PATH",
    os.path.join(ESP_ROOT, "socs/defconfig", f"esp_{ESP_BOARD}_defconfig"),
)
SIM_TIMEOUT = int(os.environ.get("ESP_SIM_TIMEOUT", str(7200)))
PASS_PATTERN = os.environ.get("ESP_SIM_PASS_PATTERN", "Program Completed") or None
SIM_INPUT = os.environ.get("ESP_SIM_INPUT") or None


def _tail(label: str, text: str, n: int = 3000) -> None:
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
    print(f"esp root : {ESP_ROOT}\nboard    : {ESP_BOARD}\ncpu      : {ESP_CPU}")

    # The sim member demands esp_xcelium, so the bundle must include a seat.
    with EspWorkspaceNode(
            reserve_bundle={"CPU": 1, "esp": 1, "esp_xcelium": 1}) as ws:
        print("\n=== stage 1: headless socgen (make esp-config) ===")
        cfg = get(ws.configure.chia_remote(
            ESP_ROOT, ESP_BOARD, esp_config_path=ESP_CONFIG_PATH))
        print(f"configure : success={cfg.success} rc={cfg.returncode}")
        if not cfg.success:
            _tail("esp-config stderr", cfg.stderr)
            ok = False

        if ok:
            print("\n=== stage 2: bare-metal software (make soft) ===")
            soft = get(ws.build.chia_remote(
                ESP_ROOT, ESP_BOARD, cpu=ESP_CPU, target="soft"))
            print(f"soft      : success={soft.success} rc={soft.returncode} "
                  f"missing={soft.missing}")
            if not soft.success:
                _tail("soft stderr", soft.stderr)
                ok = False

        if ok:
            print(f"\n=== stage 3: RTL simulation (make xmsim, "
                  f"timeout={SIM_TIMEOUT}s) ===")
            sim = get(ws.sim.chia_remote(
                ESP_ROOT, ESP_BOARD, sim_input=SIM_INPUT,
                pass_pattern=PASS_PATTERN, timeout_seconds=SIM_TIMEOUT))
            print(f"sim       : success={sim.success} rc={sim.returncode} "
                  f"pass_matched={sim.pass_matched}")
            _tail("xmsim stdout", sim.stdout)
            if not sim.success:
                _tail("xmsim stderr", sim.stderr)
                ok = False

    print("\n" + ("PASS: socgen + soft + xmsim succeeded on one ESP workspace."
                  if ok else "FAIL: see the markers above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
