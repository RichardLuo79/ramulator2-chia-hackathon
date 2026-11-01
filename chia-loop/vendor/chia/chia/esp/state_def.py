"""chia.esp.state_def — result/artifact dataclasses for the ESP nodes.

ESP (Embedded Scalable Platforms, Columbia SLD; github.com/sld-columbia/esp)
drives everything through ``make`` targets run from a per-board working
directory ``<esp_root>/socs/<board>/``, and that directory is the state:
config, generated RTL, and software outputs accumulate in place, with later
targets reading what earlier ones wrote. These artifacts therefore mostly
describe what happened on the worker; small binaries ship by value and
anything over a caller-set cap stays in the workspace.
"""

from dataclasses import dataclass, field
from typing import Literal


# CPU tiles ESP can instantiate; names double as the ``soft-build/<cpu>``
# output subdirectory. (ariane = 64-bit RISC-V CVA6, ibex = 32-bit RISC-V,
# leon3 = 32-bit SPARC V8.)
EspCpu = Literal["ariane", "ibex", "leon3"]

# Software build flavors, mapping 1:1 onto the ``make soft`` / ``make linux``
# targets in ``socs/<board>/``.
EspSoftTarget = Literal["soft", "linux"]


@dataclass
class EspMakeResult:
    """One generic ESP make-target run."""
    success: bool
    returncode: int    # -1 on timeout
    target: str        # e.g. "esp-config", "soft", "sim"
    work_dir: str      # directory the target ran in, on the worker
    stdout: str
    stderr: str
    # Manifest (relpath -> size in bytes) of the list_dir passed to make();
    # empty when none was given. Contents stay on the worker.
    listing: dict[str, int] = field(default_factory=dict)


@dataclass
class EspCollectResult:
    """Text files fetched by value from a workspace."""
    base_dir: str
    files: dict[str, str]      # relpath -> contents (errors="replace")
    skipped: dict[str, int]    # matched but over max_bytes_per_file; size shown
    listing: dict[str, int]    # fresh manifest of base_dir at collect time


@dataclass
class EspConfigResult:
    """One headless socgen run (``make esp-config``)."""
    success: bool
    returncode: int    # -1 on timeout
    board_dir: str     # the socs/<board> dir that was configured
    # Full text of the .esp_config used — small and sufficient to regenerate
    # the SoC; carry it alongside downstream artifacts for reproducibility.
    esp_config: str
    stdout: str
    stderr: str


# Accelerator design flows accgen can generate. "rtl" is license-free; the
# HLS flows need their tool on PATH — "vivado" (Vivado HLS, C++) ships with
# Vivado through 2020.1, "stratus" (Stratus HLS, SystemC) is Cadence's
# bdw_* toolchain, "catapult" (Siemens Catapult HLS) generates SystemC/
# MatchLib skeletons. ESP also documents hand-written C++ Catapult
# accelerators (<name>_cxx_catapult); those never go through accgen, but
# the name-keyed workspace members drive their make targets all the same.
# Verify your site's license entitlement before using an HLS flow.
# hls4ml is deferred (its accgen branch needs an external project dir).
EspAccelFlow = Literal["rtl", "vivado", "stratus", "catapult"]

_FLOW_ANSWER = {"rtl": "R", "vivado": "V", "stratus": "S", "catapult": "C"}

# Skeleton directory under accelerators/ per flow (accgen's FLOW_DIR).
_FLOW_DIR = {
    "rtl": "rtl",
    "vivado": "vivado_hls",
    "stratus": "stratus_hls",
    "catapult": "catapult_hls",
}

# Skeleton-name suffix per flow (accgen's NAMEFULL): catapult inserts its
# hardcoded SystemC language tag, the others use the flow name alone.
_FLOW_SUFFIX = {
    "rtl": "rtl",
    "vivado": "vivado",
    "stratus": "stratus",
    "catapult": "sysc_catapult",
}


@dataclass
class EspAccelSpec:
    """Inputs for one accgen.sh run (the script's prompts, in order)."""
    name: str
    flow: EspAccelFlow = "rtl"
    device_id: str = ""              # three hex digits; "" = accgen's default
    # Answers for the remaining prompts (registers, bit-width, data sizes,
    # chunking, batching). Empty strings take accgen's defaults; the default
    # tail is long enough that trailing extras are ignored harmlessly.
    answers_tail: list[str] = field(default_factory=lambda: [""] * 16)

    @property
    def make_name(self) -> str:
        """Name in per-accelerator make targets (``<make_name>-hls``, ...):
        the skeleton directory's basename, not the bare accelerator name."""
        return f"{self.name}_{_FLOW_SUFFIX[self.flow]}"

    @property
    def acc_dir(self) -> str:
        """Skeleton directory relative to the ESP root."""
        return f"accelerators/{_FLOW_DIR[self.flow]}/{self.make_name}"

    def to_answers(self) -> str:
        """Render the newline-separated stdin script accgen.sh consumes."""
        if self.flow not in _FLOW_ANSWER:
            raise ValueError(
                f"flow must be one of {sorted(_FLOW_ANSWER)} (got {self.flow!r})"
            )
        head = [self.name, _FLOW_ANSWER[self.flow], "", self.device_id]
        return "\n".join(head + list(self.answers_tail)) + "\n"


@dataclass
class EspAccgenResult:
    """One accgen.sh run: a generated accelerator skeleton."""
    success: bool              # accgen exited 0 and the skeleton dir exists
    returncode: int            # -1 on timeout
    acc_dir: str               # worker-side absolute path of the skeleton
    listing: dict[str, int]    # relpath -> size of every generated file
    stdout: str
    stderr: str


@dataclass
class EspAccelResult:
    """One per-accelerator make target (``<name>-hls``, ``<name>-baremetal``, ...)."""
    success: bool
    returncode: int    # -1 on timeout
    name: str          # accelerator name the target was invoked for
    action: str        # e.g. "hls", "baremetal"
    stdout: str
    stderr: str


@dataclass
class EspSimResult:
    """One full-system RTL simulation (``make xmsim``)."""
    success: bool              # make exited 0 (and pass_pattern matched, when given)
    returncode: int            # -1 on timeout
    board_dir: str             # the socs/<board> dir the sim ran in
    test_program: str | None   # TEST_PROGRAM the run used; None = default systest
    pass_matched: bool | None  # pass_pattern found in stdout; None when no pattern given
    stdout: str                # simulator console transcript
    stderr: str


@dataclass
class EspSynthResult:
    """One FPGA synthesis run (``make vivado-syn``)."""
    success: bool              # make exited 0 AND the bitstream exists
    returncode: int            # -1 on timeout
    board_dir: str
    bitstream: str | None      # worker-side path; stays in the workspace
    reports: dict[str, str]    # relpath -> text of the implementation reports
    stdout: str
    stderr: str


@dataclass
class EspFpgaRunResult:
    """One software run on the programmed FPGA (``make fpga-run[-linux]``)
    with the UART console captured over TCP."""
    success: bool              # make exited 0 (and pass_pattern matched, when given)
    returncode: int            # -1 on timeout
    board_dir: str
    pass_matched: bool | None  # pass_pattern seen on the UART; None when no pattern
    uart: str                  # console transcript
    stdout: str                # make/esplink output
    stderr: str


@dataclass
class EspSoftArtifact:
    """ESP software build outputs (``make soft`` / ``make linux``) for one SoC.

    Outputs land in ``<board_dir>/soft-build/<cpu>/`` on the worker; files at
    or under the build's inline cap ship by value in ``binaries``, larger
    ones (normally just ``linux.bin``) are recorded in ``kept`` and stay in
    the workspace.
    """
    target: EspSoftTarget
    cpu: str                    # names the soft-build/<cpu> output subdir
    board: str
    success: bool               # make exited 0 AND every expected output exists
    binaries: dict[str, bytes]  # filename -> bytes for outputs at/under the cap
    kept: dict[str, int]        # filename -> size for outputs left on the worker
    soft_build_dir: str         # worker-side base dir of binaries and kept
    missing: list[str]          # expected outputs absent after the build
    stdout: str
    stderr: str
    returncode: int             # -1 on timeout
