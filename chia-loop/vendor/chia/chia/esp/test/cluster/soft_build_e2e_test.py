"""End-to-end test for the license-free ESP flow on a local ESP cluster.

Exercises the license-free ESP flow against the REAL ESP tools inside a
``ghcr.io/ucb-bar/chia-esp`` worker container (cluster up via
esp_local.yaml): headless SoC configuration (socgen) followed by the
bare-metal and Linux software builds, all on ONE EspWorkspaceNode bundle:

    ws.configure        (.esp_config -> make esp-config)
      -> ws.build("soft")   (prom.bin + systest.bin, by value)
      -> ws.build("linux")  (linux.bin)
      -> ws.collect         (round-trips .esp_config; proves the size cap)

No CAD licenses are needed anywhere in this test.

Usage (host, chia env active, cluster up via esp_cluster.sh):
    cd <repo root>
    python chia/esp/test/cluster/soft_build_e2e_test.py

Environment overrides:
    ESP_ROOT         ESP checkout inside the worker (default /home/espuser/esp)
    ESP_BOARD        socs/<board> to configure (default xilinx-vc707-xc7vx485t)
    ESP_CPU          processor tile in the config (default ariane, matching
                     the default config below)
    ESP_CONFIG_PATH  worker-side saved config to copy in (default: the
                     board's defconfig, socs/defconfig/esp_<board>_defconfig.
                     The saved example configs under soft/common/apps/
                     examples/ predate the current socgen parser — e.g.
                     multifft's lacks the NCPU_TILE line — so the defconfig
                     is the reliable headless source at ESP HEAD)
    ESP_E2E_SKIP_LINUX=1   skip the (hours-long) `make linux` stage
    ESP_LINUX_TIMEOUT      seconds for `make linux` (default 14400)
"""

import os
import sys

# Allow `python .../soft_build_e2e_test.py` from any cwd: put the repo root
# on sys.path so the `chia` namespace package imports.
# cluster -> test -> esp -> chia -> <repo root> == 4 levels up.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 4)))
sys.path.insert(0, _REPO_ROOT)

import ray

from chia.base.ChiaFunction import get
from chia.esp.esp_workspace import EspWorkspaceNode, board_dir

ESP_ROOT = os.environ.get("ESP_ROOT", "/home/espuser/esp")
ESP_BOARD = os.environ.get("ESP_BOARD", "xilinx-vc707-xc7vx485t")
ESP_CPU = os.environ.get("ESP_CPU", "ariane")
ESP_CONFIG_PATH = os.environ.get(
    "ESP_CONFIG_PATH",
    os.path.join(ESP_ROOT, "socs/defconfig", f"esp_{ESP_BOARD}_defconfig"),
)
SKIP_LINUX = os.environ.get("ESP_E2E_SKIP_LINUX", "") == "1"
LINUX_TIMEOUT = int(os.environ.get("ESP_LINUX_TIMEOUT", str(4 * 3600)))


def _tail(label: str, text: str, n: int = 1200) -> None:
    print(f"--- {label} (tail) ---\n{text[-n:] if text else '(empty)'}\n---")


def main() -> int:
    print(f"[driver] connecting to ray cluster (working_dir={_REPO_ROOT})")
    # Ship the live repo as the worker working_dir: the chia-esp image's
    # installed chia may predate chia.esp, so workers must import the source
    # from here.
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
    print(f"esp root  : {ESP_ROOT}")
    print(f"board dir : {bdir}")
    print(f"cpu       : {ESP_CPU}")
    print(f"config    : {ESP_CONFIG_PATH}")

    # One workspace bundle: every stage lands on the same worker.
    with EspWorkspaceNode() as ws:
        print("\n=== stage 1: headless socgen (make esp-config) ===")
        cfg = get(ws.configure.chia_remote(
            ESP_ROOT, ESP_BOARD, esp_config_path=ESP_CONFIG_PATH))
        print(f"configure : success={cfg.success} rc={cfg.returncode} "
              f"config_chars={len(cfg.esp_config)}")
        if not cfg.success:
            _tail("esp-config stderr", cfg.stderr)
            _tail("esp-config stdout", cfg.stdout)
            ok = False

        # .esp_config round-trip through collect proves put/collect plumbing
        # and that configure wrote what it echoed.
        col = get(ws.collect.chia_remote(bdir, ["socgen/esp/.esp_config"]))
        roundtrip_ok = col.files.get("socgen/esp/.esp_config") == cfg.esp_config
        print(f"collect   : .esp_config round-trip ok={roundtrip_ok}")
        ok = ok and roundtrip_ok

        if ok:
            print("\n=== stage 2: bare-metal software (make soft) ===")
            soft = get(ws.build.chia_remote(
                ESP_ROOT, ESP_BOARD, cpu=ESP_CPU, target="soft"))
            print(f"soft      : success={soft.success} rc={soft.returncode} "
                  f"binaries={ {n: len(b) for n, b in soft.binaries.items()} } "
                  f"kept={soft.kept} missing={soft.missing}")
            if not soft.success:
                _tail("soft stderr", soft.stderr)
                ok = False
            elif not all(soft.binaries.get(n) for n in ("prom.bin", "systest.bin")):
                print("  ! expected non-empty prom.bin and systest.bin by value")
                ok = False

        if ok and not SKIP_LINUX:
            print(f"\n=== stage 3: linux (make linux, timeout={LINUX_TIMEOUT}s) ===")
            linux = get(ws.build.chia_remote(
                ESP_ROOT, ESP_BOARD, cpu=ESP_CPU, target="linux",
                timeout_seconds=LINUX_TIMEOUT))
            size = linux.kept.get("linux.bin", len(linux.binaries.get("linux.bin", b"")))
            print(f"linux     : success={linux.success} rc={linux.returncode} "
                  f"linux.bin={size} bytes "
                  f"({'kept on worker' if 'linux.bin' in linux.kept else 'inline'}) "
                  f"at {linux.soft_build_dir}")
            if not linux.success or size == 0:
                _tail("linux stderr", linux.stderr)
                ok = False
        elif SKIP_LINUX:
            print("\n(stage 3 skipped: ESP_E2E_SKIP_LINUX=1)")

        if ok:
            # Cap-proof: collect the built binaries with a tiny cap — they must
            # land in `skipped` (sizes), not ship through the object store.
            col = get(ws.collect.chia_remote(
                bdir, ["soft-build/**/*.bin"], max_bytes_per_file=1024))
            print(f"\ncollect cap: skipped={col.skipped} (expected: every .bin)")
            if not col.skipped or col.files:
                print("  ! expected all .bin outputs to be size-capped into `skipped`")
                ok = False

    print("\n" + ("PASS: socgen + software builds succeeded on one ESP workspace."
                  if ok else "FAIL: see the markers above."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
