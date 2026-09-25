"""End-to-end test for the ESP FPGA flow: synthesize, program, run.

Runs the full hardware chain on ONE EspWorkspaceNode bundle (which must
include the ``esp_vivado`` and ``esp_fpga`` seats):

    ws.configure  (board defconfig -> make esp-config)
      -> ws.build("soft")     (bare-metal systest + boot images)
      -> ws.synth             (make vivado-syn: hours; bitstream + reports)
      -> ws.fpga_program      (bitstream -> board, via hw_server)
      -> ws.fpga_run          (esplink loads + starts systest; the UART
                               transcript is the verdict)

The hw_server / UART / EDCL endpoints come from the site env (see
esp_site_env.example.sh) — typically ssh -L tunnels that make the FPGA
host's services look local to the worker.

NOTE: ESP's per-board scripts pin exact Xilinx IP versions
(``constraints/<board>/*.tcl``), and a Vivado release whose IP catalog has
moved on fails synthesis at IP creation. When that happens, adjust the
pinned version in the workspace's copy (or synthesize with an era-matched
Vivado) — the right value depends on your tool version, so no single patch
ships here.

Usage (host, chia env active, cluster up via esp_xcelium.yaml):
    cd <repo root>
    python chia/esp/test/cluster/fpga_e2e_test.py

Environment overrides:
    ESP_ROOT / ESP_CPU              as in the other drivers
    ESP_BOARD                       default xilinx-vcu118-xcvu9p
    ESP_SYN_TIMEOUT                 seconds for make vivado-syn (default 6h)
    ESP_FPGA_PASS_PATTERN           UART regex for success (default
                                    "Hello from ESP" — what systest prints)
    ESP_UART_TIMEOUT                seconds to watch the console (default 600)
    ESP_E2E_SKIP_SYNTH=1            reuse the workspace's existing bitstream
    CHIA_ESP_FPGA_HOST / CHIA_ESP_HW_SERVER_PORT / CHIA_ESP_UART_HOST /
    CHIA_ESP_UART_PORT / CHIA_ESP_ESPLINK_IP / CHIA_ESP_VIVADO_PROG_BIN
                                    hardware endpoints (site env)
"""

import os
import sys
import time

# cluster -> test -> esp -> chia -> <repo root> == 4 levels up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 4)))
sys.path.insert(0, _REPO_ROOT)

import ray

from chia.base.ChiaFunction import get
from chia.esp.esp_workspace import EspWorkspaceNode

# The workspace must be reachable at the SAME path from the container
# (configure/build) and the head's bare Vivado worker (synth). When a shared
# path is configured, use it as esp_root everywhere; else fall back to the
# in-container checkout (single-worker setups where synth also runs there).
ESP_ROOT = (os.environ.get("CHIA_ESP_SHARED_WORKSPACE")
            or os.environ.get("ESP_ROOT", "/home/espuser/esp"))
ESP_BOARD = os.environ.get("ESP_BOARD", "xilinx-vcu118-xcvu9p")
ESP_CPU = os.environ.get("ESP_CPU", "ariane")
SYN_TIMEOUT = int(os.environ.get("ESP_SYN_TIMEOUT", str(6 * 3600)))
PASS_PATTERN = os.environ.get("ESP_FPGA_PASS_PATTERN", "Hello from ESP")
UART_TIMEOUT = int(os.environ.get("ESP_UART_TIMEOUT", "600"))
SKIP_SYNTH = os.environ.get("ESP_E2E_SKIP_SYNTH", "") == "1"

FPGA_HOST = os.environ.get("CHIA_ESP_FPGA_HOST", "127.0.0.1")
HW_SERVER_PORT = int(os.environ.get("CHIA_ESP_HW_SERVER_PORT", "3121"))
UART_HOST = os.environ.get("CHIA_ESP_UART_HOST", "127.0.0.1")
UART_PORT = int(os.environ.get("CHIA_ESP_UART_PORT", "4001"))
ESPLINK_IP = os.environ.get("CHIA_ESP_ESPLINK_IP", "127.0.0.1")
VIVADO_PROG_BIN = os.environ.get("CHIA_ESP_VIVADO_PROG_BIN") or None
VIVADO_SYN_BIN = os.environ.get("CHIA_ESP_VIVADO_SYN_BIN") or None

DEFCONFIG = f"socs/defconfig/esp_{ESP_BOARD}_defconfig"


def _tail(label: str, text: str, n: int = 2000) -> None:
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
    # The colocated bundle carries the container-side members (configure /
    # build / accel / fpga_program / fpga_run). synth is dispatched OFF this
    # bundle, to the esp_vivado seat on the head (see below).
    with EspWorkspaceNode(reserve_bundle={
            "CPU": 1, "esp": 1, "esp_fpga": 1}) as ws:

        print("\n=== stage 1: configure + soft ===")
        cfg = get(ws.configure.chia_remote(
            ESP_ROOT, ESP_BOARD,
            esp_config_path=os.path.join(ESP_ROOT, DEFCONFIG)))
        soft = get(ws.build.chia_remote(ESP_ROOT, ESP_BOARD, cpu=ESP_CPU))
        print(f"configure: success={cfg.success}  soft: success={soft.success}")
        if not (cfg.success and soft.success):
            _tail("configure stderr", cfg.stderr)
            _tail("soft stderr", soft.stderr)
            ok = False

        if ok and not SKIP_SYNTH:
            print(f"\n=== stage 2: vivado-syn (timeout={SYN_TIMEOUT}s) ===")
            t0 = time.time()
            # Unpinned dispatch: synth demands only esp_vivado, so it lands on
            # the head's bare Vivado worker rather than the container bundle.
            # It reads/writes the SAME workspace (shared same-path mount), so
            # its physical worker need not be the container's.
            syn = get(EspWorkspaceNode.synth.chia_remote(
                ESP_ROOT, ESP_BOARD, vivado_bin=VIVADO_SYN_BIN,
                timeout_seconds=SYN_TIMEOUT))
            print(f"synth    : success={syn.success} rc={syn.returncode} "
                  f"({time.time() - t0:.0f}s)")
            print(f"bitstream: {syn.bitstream}")
            print(f"reports  : {sorted(syn.reports)}")
            for name, text in syn.reports.items():
                if "timing_summary" in name:
                    _tail(name, text, n=1500)
            if not syn.success:
                _tail("synth stdout", syn.stdout, n=4000)
                _tail("synth stderr", syn.stderr)
                ok = False
        elif SKIP_SYNTH:
            print("\n=== stage 2: SKIPPED (reusing workspace bitstream) ===")

        if ok:
            print(f"\n=== stage 3: fpga-program via {FPGA_HOST}:{HW_SERVER_PORT} ===")
            prog = get(ws.fpga_program.chia_remote(
                ESP_ROOT, ESP_BOARD, fpga_host=FPGA_HOST,
                hw_server_port=HW_SERVER_PORT, vivado_bin=VIVADO_PROG_BIN))
            print(f"program  : success={prog.success} rc={prog.returncode}")
            if not prog.success:
                _tail("program stdout", prog.stdout)
                _tail("program stderr", prog.stderr)
                ok = False

        if ok:
            print(f"\n=== stage 4: fpga-run (UART {UART_HOST}:{UART_PORT}, "
                  f"pattern {PASS_PATTERN!r}) ===")
            run = get(ws.fpga_run.chia_remote(
                ESP_ROOT, ESP_BOARD, uart_host=UART_HOST, uart_port=UART_PORT,
                esplink_ip=ESPLINK_IP, pass_pattern=PASS_PATTERN,
                uart_timeout_seconds=UART_TIMEOUT))
            print(f"run      : success={run.success} rc={run.returncode} "
                  f"pass_matched={run.pass_matched}")
            _tail("UART transcript", run.uart, n=3000)
            if not run.success:
                _tail("run stdout", run.stdout)
                _tail("run stderr", run.stderr)
                ok = False

    print("\n" + ("PASS: the SoC was synthesized, programmed, and ran on the "
                  "FPGA — console output confirmed over UART."
                  if ok else "FAIL: see the markers above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
