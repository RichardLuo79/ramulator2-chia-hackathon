"""chia.esp.esp_workspace — one node class for ESP make flows.

ESP exposes every flow as a ``make`` target run from a working directory —
``<esp_root>/socs/<board>/`` for full-SoC targets, an ``accelerators/``
subdir for per-accelerator ones. That directory is PATH-BASED state on the
worker: later targets read what earlier ones wrote, so chained targets
(esp-config -> soft -> sim) and file fetches must land on the SAME worker.

:class:`EspWorkspaceNode` models exactly that: one instance = one workspace,
pinned to one placement-group bundle. Its members are all stateless
per-call functions — generic primitives (``make`` / ``put_file`` /
``remove`` / ``collect``) plus one typed member per ESP flow
(``configure``, ``build``, ``sim``, ``accgen``, ``accel``; FPGA members to
come). ``EspWorkspaceNode.<fn>.chia_remote`` (the class attribute) is the
raw, unpinned form for callers that handle placement themselves.

Workers are assumed ESP-ready (toolchains on PATH); use the ``env`` argument
for per-call overrides.
"""

import glob as _glob
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading

from chia.base.ChiaFunction import ChiaFunction
from chia.base.colocated import ColocatedNode
from chia.esp.state_def import (
    EspAccelResult,
    EspAccelSpec,
    EspAccgenResult,
    EspCollectResult,
    EspConfigResult,
    EspFpgaRunResult,
    EspMakeResult,
    EspSimResult,
    EspSoftArtifact,
    EspSoftTarget,
    EspSynthResult,
)

logger = logging.getLogger(__name__)

# The board Makefile's ESP_CFG_BUILD: where socgen reads .esp_config from and
# writes socmap.vhd / esplink.h to, relative to the board dir.
ESP_CFG_BUILD = "socgen/esp"

# Canonical outputs of `build` per target, relative to soft-build/<cpu>/.
_TARGET_OUTPUTS: dict[str, tuple[str, ...]] = {
    "soft": ("prom.bin", "systest.bin"),
    "linux": ("linux.bin",),
}

# Batch input script for `make xmsim`. ESP's own auto-generated xcelium/
# xmsim.in sets only severity options — without `run`/`exit` the simulator
# stops at its console prompt — so `sim` pre-writes this superset (the file
# is a no-prerequisite make target: an existing one is never regenerated).
XMSIM_BATCH_INPUT = """\
set severity_pack_assert_off {warning}
set pack_assert_off { std_logic_arith numeric_std }
set intovf_severity_level {ignore}
run
exit
"""


def board_dir(esp_root: str, board: str) -> str:
    """Absolute path of the per-board working directory ``socs/<board>``."""
    return os.path.join(os.path.abspath(esp_root), "socs", board)


def _list_files(base_dir: str) -> dict[str, int]:
    """Manifest of every file under base_dir: relative path -> size in bytes."""
    listing: dict[str, int] = {}
    for root, _dirs, names in os.walk(base_dir):
        for name in names:
            path = os.path.join(root, name)
            try:
                listing[os.path.relpath(path, base_dir)] = os.path.getsize(path)
            except OSError:
                pass  # dangling symlink etc.
    return listing


def _tool_env(env: dict[str, str] | None) -> dict[str, str]:
    """Environment for ESP tool subprocesses.

    The worker's own environment, minus this interpreter's bin dir on PATH:
    ESP's Makefiles invoke ``python3`` and must resolve the image's python
    (which has ESP's deps), not the venv running this node. Caller ``env``
    entries are layered on last.
    """
    merged = dict(os.environ)
    own_bin = os.path.dirname(sys.executable)
    merged["PATH"] = os.pathsep.join(
        p for p in merged.get("PATH", "").split(os.pathsep) if p and p != own_bin
    )
    if env:
        merged.update(env)
    return merged


def _run_proc(
    cmd: list[str],
    work_dir: str,
    env: dict[str, str] | None = None,
    timeout_seconds: int = 86400,
    stdin_text: str | None = None,
) -> tuple[str, str, int]:
    """Run one ESP tool subprocess; never raises.

    Returns (stdout, stderr, returncode), with ``returncode=-1`` on timeout.
    """
    work_dir = os.path.abspath(work_dir)
    logger.info(f"Running: {' '.join(cmd)} (cwd={work_dir})")

    # cwd AND the PWD env var must both point at work_dir: ESP's Makefiles
    # derive output/include paths from $(PWD), which is inherited from the
    # environment and not updated by cwd= (or by make -C).
    # start_new_session puts the whole tool tree in one process group;
    # chia's pid_registry tracks the pgid so chia_cancel() can kill it.
    # With no stdin_text, stdin is closed so no tool can silently wait on
    # console input.
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=work_dir,
            start_new_session=True,
            env=_tool_env({"PWD": work_dir, **(env or {})}),
        )
    except OSError as e:
        logger.error(f"could not launch {' '.join(cmd)}: {e}")
        return "", str(e), -1
    returncode: int
    try:
        stdout, stderr = proc.communicate(input=stdin_text, timeout=timeout_seconds)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        stderr += f"\n{cmd[0]} timed out after {timeout_seconds}s"
        logger.error(f"{' '.join(cmd)} timed out after {timeout_seconds}s")
        returncode = -1

    if returncode != 0:
        logger.error(f"{' '.join(cmd)} failed (rc={returncode}); "
                     f"stderr tail: {stderr[-500:] if stderr else '(empty)'}")
    return stdout, stderr, returncode


def _run_make(
    work_dir: str,
    target: str,
    make_vars: dict[str, str] | None = None,
    jobs: int = 1,
    env: dict[str, str] | None = None,
    timeout_seconds: int = 86400,
    stdin_text: str | None = None,
) -> tuple[str, str, int]:
    """Run one ``make <vars> <target>`` in ``work_dir``; never raises."""
    cmd = ["make", f"-j{jobs}"]
    for key, value in (make_vars or {}).items():
        cmd.append(f"{key}={value}")
    cmd.append(target)
    return _run_proc(cmd, work_dir, env=env, timeout_seconds=timeout_seconds,
                     stdin_text=stdin_text)


def with_acc_tile(
    config_text: str,
    acc: str,
    row: int,
    col: int,
    impl: str = "basic_dma64",
    has_l2: int = 0,
    vendor: str = "sld",
) -> str:
    """Return ``config_text`` with tile (row, col) replaced by an accelerator.

    Rewrites the ``TILE_<row>_<col>`` line (keeping its tile index) to
    ``... acc <ACC> 0 0 0 <impl> <has_l2> <vendor>`` and points the matching
    ``POWER_`` line's first field at the accelerator. socgen identifies
    accelerators by their tech-library subdirectory name UPPERCASED (an
    unknown name silently becomes an empty tile), so ``acc`` should be the
    make name (e.g. "chiatest_rtl") and is uppercased here. ``impl`` must be
    an implementation point whose DMA width matches the SoC's (64 for the
    RISC-V CPUs), or socgen filters it out.

    Raises ValueError when the tile or power line is absent.
    """
    acc_token = acc.upper()
    tile_re = re.compile(rf"^TILE_{row}_{col} = (\d+) .*$", re.MULTILINE)
    m = tile_re.search(config_text)
    if m is None:
        raise ValueError(f"no TILE_{row}_{col} line in config")
    config_text = tile_re.sub(
        f"TILE_{row}_{col} = {m.group(1)} acc {acc_token} 0 0 0 "
        f"{impl} {has_l2} {vendor}",
        config_text,
    )
    power_re = re.compile(rf"^POWER_{row}_{col} = \S+ (.*)$", re.MULTILINE)
    if power_re.search(config_text) is None:
        raise ValueError(f"no POWER_{row}_{col} line in config")
    return power_re.sub(rf"POWER_{row}_{col} = {acc_token} \1", config_text)


def _seed_xcelium_cds_defaults(esp_root: str, env: dict[str, str]) -> None:
    """Map STD/IEEE for the Xcelium VHDL compiles ESP's xmsim flow runs.

    Vivado's ``compile_simlib`` (and the cds.lib ESP copies from it into the
    board workspace) never includes the Xcelium install's default library
    mappings, so every VHDL compile dies with ``NOLSTD`` in a clean batch
    environment. Prepending a ``softinclude`` of the install's own cds.lib to
    ``<esp_root>/.cache/xcelium/cds.lib`` fixes both layers; the file
    survives compile_simlib reruns. The install root is derived from the
    ``xmvhdl`` on the tool PATH, so nothing here is site-specific. No-op
    (with a warning) when xmvhdl or the install's cds.lib can't be found,
    and when the include line is already present.
    """
    xmvhdl = shutil.which("xmvhdl", path=env.get("PATH"))
    if xmvhdl is None:
        logger.warning("xmvhdl not on the tool PATH; skipping cds.lib seeding")
        return
    root = os.path.dirname(xmvhdl)
    default_cds = None
    for _ in range(5):
        root = os.path.dirname(root)
        candidate = os.path.join(root, "tools", "xcelium", "files", "cds.lib")
        if os.path.isfile(candidate):
            default_cds = candidate
            break
    if default_cds is None:
        logger.warning(f"no default cds.lib found above {xmvhdl}; skipping seeding")
        return

    include_line = f"softinclude {default_cds}"
    cache_cds = os.path.join(os.path.abspath(esp_root), ".cache", "xcelium", "cds.lib")
    existing = ""
    if os.path.isfile(cache_cds):
        with open(cache_cds, errors="replace") as f:
            existing = f.read()
        if include_line in existing:
            return
    os.makedirs(os.path.dirname(cache_cds), exist_ok=True)
    with open(cache_cds, "w") as f:
        f.write(include_line + "\n" + existing)
    logger.info(f"Seeded {cache_cds} with {include_line!r}")


def _resolve_under(base_dir: str, relpath: str) -> str:
    """Absolute path of ``relpath`` under ``base_dir``; ValueError on escape."""
    base_dir = os.path.abspath(base_dir)
    path = os.path.normpath(os.path.join(base_dir, relpath))
    if path != base_dir and not path.startswith(base_dir + os.sep):
        raise ValueError(
            f"relpath {relpath!r} escapes base dir {base_dir!r} (-> {path!r})"
        )
    return path


class EspWorkspaceNode(ColocatedNode):
    """One ESP workspace: all its flow members share one placement.

    The members are ``@staticmethod @ChiaFunction(resources={"esp": 1})``;
    ``__init__`` re-binds each into a per-instance pinned form so
    ``node.<fn>.chia_remote(...)`` lands on this node's bundle::

        with EspWorkspaceNode() as ws:        # reserves {"CPU": 1, "esp": 1}
            root, board = "/home/espuser/esp", "xilinx-vc707-xc7vx485t"
            cfg = get(ws.configure.chia_remote(
                root, board,
                esp_config_path=f"{root}/socs/defconfig/esp_{board}_defconfig"))
            soft = get(ws.build.chia_remote(root, board, cpu="ariane"))
            outs = get(ws.collect.chia_remote(
                board_dir(root, board), ["soft-build/**/*.log"]))
    """

    _MEMBER_FNS = ("make", "put_file", "remove", "collect", "configure",
                   "build", "sim", "accgen", "accel", "synth",
                   "fpga_program", "fpga_run")
    _DEFAULT_BUNDLE = {"CPU": 1, "esp": 1}

    @staticmethod
    @ChiaFunction(resources={"esp": 1})
    def make(
        work_dir: str,
        target: str,
        make_vars: dict[str, str] | None = None,
        jobs: int = 1,
        env: dict[str, str] | None = None,
        timeout_seconds: int = 86400,
        list_dir: str | None = None,
    ) -> EspMakeResult:
        """Run one ``make -C <work_dir> <target>`` on a worker.

        The generic escape hatch behind the typed members.

        Args:
            work_dir: Directory to run make in.
            target: Any ESP Makefile target: "esp-config", "soft", "linux",
                "sim", "vivado-syn", "<acc>-hls", ...
            make_vars: Makefile variables appended to the command line, e.g.
                ``{"TEST_PROGRAM": "./soft-build/ariane/baremetal/fft.exe"}``.
            jobs: make -j level.
            env: Extra environment variables layered over the worker's.
            timeout_seconds: Wall-clock limit; the make tree is killed and
                ``returncode=-1`` returned on expiry.
            list_dir: When given, ``listing`` manifests this directory after
                the run. Opt-in because a full board dir is a large tree.
        """
        stdout, stderr, returncode = _run_make(
            work_dir, target, make_vars=make_vars, jobs=jobs, env=env,
            timeout_seconds=timeout_seconds,
        )
        return EspMakeResult(
            success=returncode == 0,
            returncode=returncode,
            target=target,
            work_dir=os.path.abspath(work_dir),
            stdout=stdout,
            stderr=stderr,
            listing=_list_files(list_dir) if list_dir is not None else {},
        )

    @staticmethod
    @ChiaFunction(resources={"esp": 1})
    def put_file(
        base_dir: str,
        relpath: str,
        content: bytes | str,
    ) -> str:
        """Write ``content`` to ``<base_dir>/<relpath>`` on this worker.

        Parent directories are created; ``relpath`` may not escape
        ``base_dir`` (ValueError). Dispatch via the pinned instance member so
        the write lands on the workspace's worker. Returns the absolute path
        written.
        """
        path = _resolve_under(base_dir, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = content.encode() if isinstance(content, str) else content
        with open(path, "wb") as f:
            f.write(data)
        logger.info(f"Wrote {len(data)} bytes to {path}")
        return path

    @staticmethod
    @ChiaFunction(resources={"esp": 1})
    def remove(
        base_dir: str,
        relpath: str,
    ) -> bool:
        """Delete ``<base_dir>/<relpath>`` (file or directory tree).

        Workspaces are stateful, and some generated state must be scrubbed
        rather than rebuilt over (e.g. stale HLS project outputs a re-run
        would silently re-install). ``relpath`` may not escape ``base_dir``
        (ValueError). Returns whether the path existed.
        """
        path = _resolve_under(base_dir, relpath)
        if not os.path.lexists(path):
            return False
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        logger.info(f"Removed {path}")
        return True

    @staticmethod
    @ChiaFunction(resources={"esp": 1})
    def collect(
        base_dir: str,
        patterns: list[str],
        max_bytes_per_file: int | None = None,
    ) -> EspCollectResult:
        """Fetch text files from a previous make's working dir on this worker.

        Dispatch via the pinned instance member so it lands on the worker
        that owns the files.

        Args:
            base_dir: Directory a previous target ran in.
            patterns: Globs relative to base_dir (``**`` is recursive), e.g.
                ``["soft-build/**/*.log", "socgen/esp/.esp_config"]``. Files
                matched by multiple patterns appear once.
            max_bytes_per_file: When set, files over this size are recorded
                in ``skipped`` instead of shipped through the object store —
                protects against a glob accidentally matching ``linux.bin``.
                ``None`` (and 0, the falsy edge) means no cap.
        """
        base_dir = os.path.abspath(base_dir)
        files: dict[str, str] = {}
        skipped: dict[str, int] = {}
        for pattern in patterns:
            for path in _glob.glob(os.path.join(base_dir, pattern), recursive=True):
                if not os.path.isfile(path):
                    continue
                rel = os.path.relpath(path, base_dir)
                if rel in files or rel in skipped:
                    continue
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if max_bytes_per_file and (size > max_bytes_per_file):
                    skipped[rel] = size
                    continue
                with open(path, errors="replace") as f:
                    files[rel] = f.read()
        if skipped:
            logger.warning(
                f"esp collect skipped {len(skipped)} file(s) over "
                f"{max_bytes_per_file} bytes: {sorted(skipped)[:5]}"
            )
        return EspCollectResult(
            base_dir=base_dir,
            files=files,
            skipped=skipped,
            listing=_list_files(base_dir),
        )

    @staticmethod
    @ChiaFunction(resources={"esp": 1})
    def configure(
        esp_root: str,
        board: str,
        esp_config: str | None = None,
        esp_config_path: str | None = None,
        make_vars: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: int = 1800,
    ) -> EspConfigResult:
        """Configure the SoC: write ``.esp_config``, run ``make esp-config``.

        Headless socgen. The config is written where the board Makefile
        actually reads it — ``socgen/esp/.esp_config`` (``ESP_CFG_BUILD``)
        under the board dir; when that file is absent ESP silently seeds it
        from the board's defconfig (``socs/defconfig/esp_<board>_defconfig``),
        ignoring any config placed elsewhere. Exactly one of ``esp_config`` /
        ``esp_config_path`` must be given.

        Args:
            esp_root: ESP checkout root on the worker.
            board: Board working-directory name under ``socs/``.
            esp_config: Full text of the ``.esp_config`` to configure with.
            esp_config_path: Worker-side path of an existing config to copy
                in instead.
            make_vars: Extra Makefile variables for the esp-config run.
            env: Extra environment variables layered over the worker's.
            timeout_seconds: Wall-clock limit; ``returncode=-1`` on expiry.

        Returns:
            EspConfigResult echoing the config text actually used.

        Raises:
            ValueError: If neither or both config sources are given.
            FileNotFoundError: If ``esp_config_path`` does not exist on the
                worker, or ``socs/<board>`` is not a directory.
        """
        if (esp_config is None) == (esp_config_path is None):
            raise ValueError(
                "exactly one of esp_config / esp_config_path must be given "
                f"(esp_config={'set' if esp_config is not None else None}, "
                f"esp_config_path={esp_config_path!r})"
            )

        bdir = board_dir(esp_root, board)
        if not os.path.isdir(bdir):
            raise FileNotFoundError(f"board dir does not exist: {bdir}")

        if esp_config_path is not None:
            with open(esp_config_path, errors="replace") as f:
                esp_config = f.read()

        config_dst = _resolve_under(bdir, os.path.join(ESP_CFG_BUILD, ".esp_config"))
        os.makedirs(os.path.dirname(config_dst), exist_ok=True)
        with open(config_dst, "w") as f:
            f.write(esp_config)
        logger.info(f"Wrote {len(esp_config)} chars to {config_dst}")

        stdout, stderr, returncode = _run_make(
            bdir, "esp-config", make_vars=make_vars, env=env,
            timeout_seconds=timeout_seconds,
        )
        return EspConfigResult(
            success=returncode == 0,
            returncode=returncode,
            board_dir=bdir,
            esp_config=esp_config,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    @ChiaFunction(resources={"esp": 1})
    def build(
        esp_root: str,
        board: str,
        cpu: str = "ariane",
        target: EspSoftTarget = "soft",
        smp: bool | None = None,
        make_vars: dict[str, str] | None = None,
        jobs: int = 16,
        env: dict[str, str] | None = None,
        timeout_seconds: int = 7200,
        inline_max_bytes: int = 16 * 1024 * 1024,
    ) -> EspSoftArtifact:
        """Build ESP software (``make soft`` / ``make linux``) and read back
        its canonical outputs.

        License-free; requires a configured SoC (the ``.esp_config`` decides
        the CPU the cross-toolchain targets), so run :meth:`configure` on
        this workspace first.

        Args:
            esp_root: ESP checkout root on the worker.
            board: Board working-directory name under ``socs/``.
            cpu: Processor tile the SoC was configured with ("ariane",
                "ibex", or "leon3") — names the ``soft-build/<cpu>`` output
                directory. If that directory is missing after a successful
                make but exactly one ``soft-build/*`` subdir exists, that one
                is used (with a warning).
            target: ``"soft"`` (bare-metal ``prom.bin`` + ``systest.bin``)
                or ``"linux"`` (``linux.bin``).
            smp: When set, passed as ``SMP=1`` / ``SMP=0`` on the make
                command line, overriding the board Makefile's setting.
                ``None`` leaves the Makefile default.
            make_vars: Any other Makefile variables; appended after ``smp``
                so they win on conflict.
            jobs: make -j level.
            env: Extra environment variables layered over the worker's.
            timeout_seconds: Wall-clock limit — ``make linux`` compiles a
                kernel + root fs, so give it hours, not minutes.
            inline_max_bytes: Outputs at or under this size ship by value in
                ``binaries``; larger ones are recorded in ``kept`` and stay
                at ``soft_build_dir`` on the worker.

        Returns:
            EspSoftArtifact; ``success`` iff make exited 0 AND every
            canonical output exists, else ``missing`` names what the build
            failed to produce.

        Raises:
            ValueError: If ``target`` is not a recognized value.
        """
        if target not in _TARGET_OUTPUTS:
            raise ValueError(
                f"target must be one of {sorted(_TARGET_OUTPUTS)} (got {target!r})"
            )

        bdir = board_dir(esp_root, board)
        all_vars: dict[str, str] = {}
        if smp is not None:
            all_vars["SMP"] = "1" if smp else "0"
        all_vars.update(make_vars or {})

        stdout, stderr, returncode = _run_make(
            bdir, target, make_vars=all_vars, jobs=jobs, env=env,
            timeout_seconds=timeout_seconds,
        )

        soft_build_dir = os.path.join(bdir, "soft-build", cpu)
        if returncode == 0 and not os.path.isdir(soft_build_dir):
            # Wrong cpu guess but an unambiguous build output: use it.
            parent = os.path.join(bdir, "soft-build")
            subdirs = [d for d in (os.listdir(parent) if os.path.isdir(parent) else [])
                       if os.path.isdir(os.path.join(parent, d))]
            if len(subdirs) == 1:
                logger.warning(
                    f"soft-build/{cpu} missing; using the only soft-build "
                    f"subdir {subdirs[0]!r} instead"
                )
                cpu = subdirs[0]
                soft_build_dir = os.path.join(parent, cpu)

        binaries: dict[str, bytes] = {}
        kept: dict[str, int] = {}
        missing: list[str] = []
        if returncode == 0:
            for name in _TARGET_OUTPUTS[target]:
                path = os.path.join(soft_build_dir, name)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    missing.append(name)
                    continue
                if size > inline_max_bytes:
                    kept[name] = size
                else:
                    with open(path, "rb") as f:
                        binaries[name] = f.read()
            if missing:
                stderr += (f"\nmake {target} exited 0 but expected outputs "
                           f"are missing from {soft_build_dir}: {missing}")
                logger.error(stderr.splitlines()[-1])

        success = returncode == 0 and not missing
        return EspSoftArtifact(
            target=target,
            cpu=cpu,
            board=board,
            success=success,
            binaries=binaries,
            kept=kept,
            soft_build_dir=soft_build_dir,
            missing=missing,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
        )

    @staticmethod
    @ChiaFunction(resources={"esp": 1, "esp_xcelium": 1})
    def sim(
        esp_root: str,
        board: str,
        test_program: str | None = None,
        sim_input: str | None = None,
        pass_pattern: str | None = None,
        clean: bool = False,
        make_vars: dict[str, str] | None = None,
        jobs: int = 1,
        env: dict[str, str] | None = None,
        timeout_seconds: int = 14400,
    ) -> EspSimResult:
        """Run a full-system RTL simulation (``make xmsim``, Xcelium).

        Compiles the configured SoC's RTL into the workspace's ``xcelium/``
        work library (incremental on repeat runs) and simulates the
        bare-metal software. Requires :meth:`configure` and a licensed
        Xcelium (plus Vivado for the one-time Xilinx simlib compile) on the
        worker; demands the ``esp_xcelium`` resource, so dispatch through a
        node whose bundle includes it, e.g.
        ``EspWorkspaceNode(reserve_bundle={"CPU": 1, "esp": 1,
        "esp_xcelium": 1})``.

        Args:
            esp_root: ESP checkout root on the worker.
            board: Board working-directory name under ``socs/``.
            test_program: Worker-side path of the bare-metal ELF to
                simulate, passed as ``TEST_PROGRAM=`` (the make dependency
                chain regenerates the boot srecs from it). ``None`` runs the
                default ``systest``.
            sim_input: Full text for ``xcelium/xmsim.in``, the simulator's
                batch input script. Defaults to ``XMSIM_BATCH_INPUT``
                (severity settings + ``run`` + ``exit``); override to bound
                the run (``run 10 ms``) or add tracing commands.
            pass_pattern: Regex searched in the console transcript; when
                given, ``success`` additionally requires a match (RTL sims
                can end with exit code 0 without the test passing).
            clean: Run ``make xmsim-distclean`` first, discarding the
                compiled work library. Needed after an SoC reconfiguration:
                Xcelium's incremental rebuild fails on units compiled
                against the previous socmap (DLCSMD checksum mismatches).
            make_vars: Any other Makefile variables.
            jobs: make -j level (RTL compile benefits).
            env: Extra environment variables layered over the worker's.
            timeout_seconds: Wall-clock limit for compile + simulate; the
                backstop for a testbench that never terminates.

        Returns:
            EspSimResult; ``success`` iff make exited 0 and, when
            ``pass_pattern`` was given, it matched.

        Raises:
            FileNotFoundError: If ``socs/<board>`` is not a directory.
        """
        bdir = board_dir(esp_root, board)
        if not os.path.isdir(bdir):
            raise FileNotFoundError(f"board dir does not exist: {bdir}")

        _seed_xcelium_cds_defaults(esp_root, _tool_env(env))

        if clean:
            _run_make(bdir, "xmsim-distclean", env=env, timeout_seconds=600)

        # After any clean: the batch input script must outlive it.
        input_dst = _resolve_under(bdir, os.path.join("xcelium", "xmsim.in"))
        os.makedirs(os.path.dirname(input_dst), exist_ok=True)
        with open(input_dst, "w") as f:
            f.write(sim_input if sim_input is not None else XMSIM_BATCH_INPUT)

        all_vars: dict[str, str] = {}
        if test_program is not None:
            all_vars["TEST_PROGRAM"] = test_program
        all_vars.update(make_vars or {})

        stdout, stderr, returncode = _run_make(
            bdir, "xmsim", make_vars=all_vars, jobs=jobs, env=env,
            timeout_seconds=timeout_seconds,
        )

        pass_matched: bool | None = None
        if pass_pattern is not None:
            pass_matched = re.search(pass_pattern, stdout) is not None
            if returncode == 0 and not pass_matched:
                logger.error(
                    f"xmsim exited 0 but pass_pattern {pass_pattern!r} not "
                    f"found in the transcript"
                )

        return EspSimResult(
            success=returncode == 0 and pass_matched is not False,
            returncode=returncode,
            board_dir=bdir,
            test_program=test_program,
            pass_matched=pass_matched,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    @ChiaFunction(resources={"esp": 1})
    def accgen(
        esp_root: str,
        spec: EspAccelSpec,
        overwrite: bool = False,
        env: dict[str, str] | None = None,
        timeout_seconds: int = 600,
    ) -> EspAccgenResult:
        """Generate an accelerator skeleton (``tools/accgen/accgen.sh``).

        accgen is an interactive prompt sequence; this drives it by piping
        ``spec.to_answers()`` on stdin (empty answers take its defaults).
        The skeleton lands in ``<esp_root>/<spec.acc_dir>``: hardware
        implementations, the accelerator XML, and bare-metal/Linux software.

        Args:
            esp_root: ESP checkout root on the worker.
            spec: The accgen prompt answers.
            overwrite: accgen refuses to regenerate an existing skeleton;
                True deletes ``spec.acc_dir`` first (idempotent reruns).
            env: Extra environment variables layered over the worker's.
            timeout_seconds: Wall-clock limit; ``returncode=-1`` on expiry.

        Returns:
            EspAccgenResult; ``success`` iff accgen exited 0 and the
            skeleton directory exists.
        """
        answers = spec.to_answers()
        esp_root = os.path.abspath(esp_root)
        if overwrite:
            shutil.rmtree(os.path.join(esp_root, spec.acc_dir), ignore_errors=True)
        stdout, stderr, returncode = _run_proc(
            ["./tools/accgen/accgen.sh"], esp_root, env=env,
            timeout_seconds=timeout_seconds, stdin_text=answers,
        )
        acc_dir = os.path.join(esp_root, spec.acc_dir)
        exists = os.path.isdir(acc_dir)
        if returncode == 0 and not exists:
            stderr += f"\naccgen exited 0 but {acc_dir} was not created"
            logger.error(stderr.splitlines()[-1])
        return EspAccgenResult(
            success=returncode == 0 and exists,
            returncode=returncode,
            acc_dir=acc_dir,
            listing=_list_files(acc_dir) if exists else {},
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    @ChiaFunction(resources={"esp": 1})
    def accel(
        esp_root: str,
        board: str,
        name: str,
        action: str,
        make_vars: dict[str, str] | None = None,
        jobs: int = 16,
        env: dict[str, str] | None = None,
        timeout_seconds: int = 3600,
    ) -> EspAccelResult:
        """Run one per-accelerator make target: ``make <name>-<action>``.

        Common actions: ``hls`` (synthesize the kernel — or, for the RTL
        flow, just package it — and install the implementation points into
        the tech library so socgen can instantiate the tile), ``baremetal``
        (the generated self-test program, output under
        ``soft-build/<cpu>/baremetal/``), ``driver``/``app`` (Linux pieces).

        Args:
            esp_root: ESP checkout root on the worker.
            board: Board working-directory name under ``socs/``.
            name: The accelerator's make name — its skeleton directory
                basename, e.g. "chiatest_rtl" (``EspAccelSpec.make_name``).
            action: Target suffix; unknown ones fail in make, not here.
            make_vars: Any other Makefile variables.
            jobs: make -j level.
            env: Extra environment variables layered over the worker's.
            timeout_seconds: Wall-clock limit; ``returncode=-1`` on expiry.
        """
        stdout, stderr, returncode = _run_make(
            board_dir(esp_root, board), f"{name}-{action}",
            make_vars=make_vars, jobs=jobs, env=env,
            timeout_seconds=timeout_seconds,
        )
        return EspAccelResult(
            success=returncode == 0,
            returncode=returncode,
            name=name,
            action=action,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    @ChiaFunction(resources={"esp_vivado": 1})
    def synth(
        esp_root: str,
        board: str,
        top: str = "top",
        overwrite_project: bool = False,
        vivado_bin: str | None = None,
        make_vars: dict[str, str] | None = None,
        jobs: int = 1,
        env: dict[str, str] | None = None,
        timeout_seconds: int = 6 * 3600,
        report_max_bytes: int = 256 * 1024,
    ) -> EspSynthResult:
        """Synthesize the configured SoC to a bitstream (``make vivado-syn``).

        A batch Vivado implementation run — hours of wall clock for the
        larger parts. The bitstream stays in the workspace (linked as
        ``<top>.bit`` in the board dir); the implementation reports Vivado
        writes alongside it ship back by value.

        Demands only ``esp_vivado`` (a Vivado seat), not ``esp`` — synthesis
        needs Vivado and the generated RTL, not the ESP cross-toolchains, so
        it can run on a Vivado-only worker (e.g. bare host) separate from the
        container that runs ``configure``/``build``. When it does, that worker
        and the container must see the workspace at the SAME path (Vivado
        writes absolute paths into its project), so pass the shared-workspace
        path as ``esp_root`` to every member.

        ESP's project-setup recipe re-runs on every invocation and asks
        interactively whether to overwrite an existing Vivado project — a
        headless run would hang on it forever, so the answer is piped in:
        "n" (reuse the project) unless ``overwrite_project``.

        Args:
            esp_root: ESP checkout root on the worker.
            board: Board working-directory name under ``socs/``.
            top: The design's top module (the board Makefile's ``TOP``),
                naming the bitstream link.
            overwrite_project: Regenerate the Vivado project instead of
                reusing it. Needed when the SoC's source list changed (e.g.
                after adding an accelerator and reconfiguring); a plain
                RTL edit does not need it. Vivado projects don't open
                across versions, so also pass it when ``vivado_bin``
                changed.
            vivado_bin: When given, a Vivado bin dir prepended to PATH for
                this run — the synthesis Vivado's IP catalog must carry the
                versions the board's scripts pin, which may rule out the
                newest install.
            make_vars: Any other Makefile variables.
            jobs: make -j level (the Vivado run manages its own threads).
            env: Extra environment variables layered over the worker's.
            timeout_seconds: Wall-clock limit; ``returncode=-1`` on expiry.
            report_max_bytes: Per-report cap; larger reports are skipped.

        Returns:
            EspSynthResult; ``success`` iff make exited 0 AND the bitstream
            exists.
        """
        bdir = board_dir(esp_root, board)
        merged_env = dict(env or {})
        if vivado_bin:
            merged_env["PATH"] = os.pathsep.join(
                [vivado_bin, merged_env.get("PATH", os.environ.get("PATH", ""))]
            )
        stdout, stderr, returncode = _run_make(
            bdir, "vivado-syn", make_vars=make_vars, jobs=jobs,
            env=merged_env, timeout_seconds=timeout_seconds,
            stdin_text="y\n" if overwrite_project else "n\n",
        )
        bitstream = os.path.join(bdir, f"{top}.bit")
        have_bit = os.path.exists(bitstream)
        if returncode == 0 and not have_bit:
            stderr += f"\nvivado-syn exited 0 but {bitstream} was not produced"
            logger.error(stderr.splitlines()[-1])
        reports: dict[str, str] = {}
        for path in _glob.glob(os.path.join(bdir, "vivado/*.runs/impl_1/*.rpt")):
            if os.path.getsize(path) <= report_max_bytes:
                with open(path, errors="replace") as f:
                    reports[os.path.relpath(path, bdir)] = f.read()
        return EspSynthResult(
            success=returncode == 0 and have_bit,
            returncode=returncode,
            board_dir=bdir,
            bitstream=bitstream if have_bit else None,
            reports=reports,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    @ChiaFunction(resources={"esp": 1, "esp_fpga": 1})
    def fpga_program(
        esp_root: str,
        board: str,
        fpga_host: str = "localhost",
        hw_server_port: int = 3121,
        vivado_bin: str | None = None,
        make_vars: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: int = 1800,
    ) -> EspMakeResult:
        """Program the FPGA with the workspace bitstream (``make fpga-program``).

        Fully network-based: connects to a Xilinx hw_server at
        ``fpga_host:hw_server_port`` (which runs on whatever machine holds
        the JTAG cable) and streams the bitstream through it.

        Args:
            esp_root: ESP checkout root on the worker.
            board: Board working-directory name under ``socs/``.
            fpga_host: Host running hw_server, as reachable from the worker.
            hw_server_port: hw_server's TCP port.
            vivado_bin: When given, a Vivado bin dir prepended to PATH for
                this run — hw_server only accepts clients at or below its
                own version, which may rule out the synthesis Vivado.
            make_vars: Any other Makefile variables.
            env: Extra environment variables layered over the worker's.
            timeout_seconds: Wall-clock limit; ``returncode=-1`` on expiry.
        """
        bdir = board_dir(esp_root, board)
        merged_env = dict(env or {})
        if vivado_bin:
            merged_env["PATH"] = os.pathsep.join(
                [vivado_bin, merged_env.get("PATH", os.environ.get("PATH", ""))]
            )
        all_vars = {"FPGA_HOST": fpga_host,
                    "XIL_HW_SERVER_PORT": str(hw_server_port),
                    **(make_vars or {})}
        stdout, stderr, returncode = _run_make(
            bdir, "fpga-program", make_vars=all_vars, env=merged_env,
            timeout_seconds=timeout_seconds,
        )
        return EspMakeResult(
            success=returncode == 0,
            returncode=returncode,
            target="fpga-program",
            work_dir=bdir,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    @ChiaFunction(resources={"esp": 1, "esp_fpga": 1})
    def fpga_run(
        esp_root: str,
        board: str,
        uart_host: str,
        uart_port: int,
        esplink_ip: str,
        linux: bool = False,
        dram_image: str | None = None,
        prom_image: str | None = None,
        pass_pattern: str | None = None,
        uart_timeout_seconds: int = 600,
        esplink_port: int | None = None,
        make_vars: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        timeout_seconds: int = 1800,
    ) -> EspFpgaRunResult:
        """Load and start software on the programmed FPGA and watch the UART.

        Loads PROM + a payload into the running SoC over its Ethernet
        (ESPLink, EDCL/UDP to ``esplink_ip``), then watches the console for a
        verdict. The board executes asynchronously, so a TCP connection to
        the UART (a ser2net- or socat-style bridge at ``uart_host:uart_port``)
        is opened BEFORE the load and read until ``pass_pattern`` matches or
        ``uart_timeout_seconds`` passes.

        By default this runs ``make fpga-run[-linux]``, which loads the
        generic ``systest``/Linux image. To run a specific program — e.g. an
        accelerator's bare-metal self-test, which the make targets can't load
        — pass ``dram_image``: esplink is built and the reset/PROM/DRAM/reset
        sequence is driven directly with that image.

        Args:
            esp_root: ESP checkout root on the worker.
            board: Board working-directory name under ``socs/``.
            uart_host: Host of the TCP-exposed UART console.
            uart_port: Its TCP port.
            esplink_ip: The SoC's EDCL IP (a socgen config field), as
                reachable from the worker.
            linux: Load ``linux.bin`` (``fpga-run-linux``) instead of the
                bare-metal ``systest.bin``. Ignored when ``dram_image`` is set.
            dram_image: Worker-side path of a program image to load into DRAM
                instead of ``systest`` — the direct-esplink path.
            prom_image: PROM image for the direct path; defaults to the sole
                ``soft-build/**/prom.bin`` in the workspace.
            pass_pattern: Regex searched in the UART transcript; when given,
                ``success`` additionally requires a match.
            uart_timeout_seconds: How long to keep reading the console after
                the load, absent a match.
            esplink_port: Override the SoC's EDCL UDP port.
            make_vars: Any other Makefile variables.
            env: Extra environment variables layered over the worker's.
            timeout_seconds: Wall-clock limit for the load.
        """
        bdir = board_dir(esp_root, board)
        try:
            sock = socket.create_connection((uart_host, uart_port), timeout=15)
        except OSError as e:
            return EspFpgaRunResult(
                success=False, returncode=-1, board_dir=bdir,
                pass_matched=False if pass_pattern else None, uart="",
                stdout="", stderr=f"could not connect to UART at "
                                  f"{uart_host}:{uart_port}: {e}",
            )
        sock.settimeout(1.0)
        collected = bytearray()
        lock = threading.Lock()
        stop_evt = threading.Event()
        found_evt = threading.Event()
        pass_re = re.compile(pass_pattern) if pass_pattern else None

        def _read_uart() -> None:
            while not stop_evt.is_set():
                try:
                    data = sock.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                with lock:
                    collected.extend(data)
                    text = collected.decode(errors="replace")
                if pass_re and pass_re.search(text):
                    found_evt.set()
                    break

        reader = threading.Thread(target=_read_uart, daemon=True)
        reader.start()

        all_vars = {"ESPLINK_IP": esplink_ip, **(make_vars or {})}
        if esplink_port is not None:
            all_vars["ESPLINK_PORT"] = str(esplink_port)

        if dram_image is not None:
            # Direct path: build esplink, then drive reset / PROM / DRAM /
            # reset ourselves so an arbitrary image (not just systest) loads.
            stdout, stderr, returncode = _run_make(
                bdir, "esplink", make_vars=all_vars, env=env,
                timeout_seconds=timeout_seconds,
            )
            if prom_image is None:
                proms = _glob.glob(
                    os.path.join(bdir, "soft-build/**/prom.bin"), recursive=True)
                prom_image = proms[0] if proms else None
            esplink = os.path.join(bdir, ESP_CFG_BUILD, "esplink")
            if returncode == 0 and prom_image and os.path.exists(esplink):
                steps = [[esplink, "--reset"],
                         [esplink, "--brom", "-i", prom_image],
                         [esplink, "--dram", "-i", dram_image],
                         [esplink, "--reset"]]
                for cmd in steps:
                    o, e, returncode = _run_proc(
                        cmd, bdir, env=env, timeout_seconds=timeout_seconds)
                    stdout += "\n" + o
                    stderr += e
                    if returncode != 0:
                        break
            elif returncode == 0:
                returncode = -1
                stderr += (f"\nesplink or prom not found "
                           f"(esplink={esplink}, prom={prom_image})")
        else:
            target = "fpga-run-linux" if linux else "fpga-run"
            stdout, stderr, returncode = _run_make(
                bdir, target, make_vars=all_vars, env=env,
                timeout_seconds=timeout_seconds,
            )

        if returncode == 0 and pass_re:
            found_evt.wait(timeout=uart_timeout_seconds)
        stop_evt.set()
        reader.join(timeout=5)
        sock.close()

        with lock:
            uart_text = collected.decode(errors="replace")
        pass_matched = None if pass_re is None else found_evt.is_set()
        if returncode == 0 and pass_matched is False:
            logger.error(f"fpga-run completed but pass_pattern "
                         f"{pass_pattern!r} not seen on the UART")
        return EspFpgaRunResult(
            success=returncode == 0 and pass_matched is not False,
            returncode=returncode,
            board_dir=bdir,
            pass_matched=pass_matched,
            uart=uart_text,
            stdout=stdout,
            stderr=stderr,
        )
