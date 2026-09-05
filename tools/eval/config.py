"""Single source of truth for the agent-loop evaluation harness.

The cycle-level oracle is GenericDDR with FRFCFS-RowHit scheduling, open rows,
refresh disabled, 64-entry read/write buffers, and 0.5/0.8 write-drain
watermarks. The candidate has the same buffer capacities but begins as a
generic fixed-delay skeleton; it intentionally has no scheduler, row policy,
or write-drain heuristic. Published immediate-response models run beside the
candidate under the same frontend, DRAM timing, trace, and metric contracts.
"""
import hashlib
import importlib
import importlib.machinery
import json
import os
import pathlib
import subprocess
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(os.environ.get("EVAL_OUT", REPO / "eval_out"))
TRACES = pathlib.Path(os.environ.get("RAMULATOR_TRACES", "/home/dev/traces"))
OPT_BUILD = pathlib.Path(
    os.environ.get("RAMULATOR_OPT_BUILD", REPO / "build-bench")
)

# ── the evaluation reference configuration (both sides) ──────────────────
REFERENCE = dict(
    read_buffer_size=64,
    write_buffer_size=64,
    wr_low_watermark=0.5,
    wr_high_watermark=0.8,
)

CANDIDATE_RESOURCES = dict(
    read_buffer_size=REFERENCE["read_buffer_size"],
    write_buffer_size=REFERENCE["write_buffer_size"],
    wr_low_watermark=REFERENCE["wr_low_watermark"],
    wr_high_watermark=REFERENCE["wr_high_watermark"],
)

MODEL_ORDER = ("candidate", "fixedlat", "md1", "wmg1", "mess")
MODEL_IMPL = {
    "candidate": "Atomic",
    "fixedlat": "FixedLat",
    "md1": "MD1",
    "wmg1": "WMG1",
    "mess": "Mess",
}
MESS_CURVES = {
    "DDR5": REPO / "tools" / "eval" / "calibration" / "mess_DDR5.txt",
}

STD = {
    "DDR5": dict(org="DDR5_16Gb_x8", timing="DDR5_4800AN", extra={}, mshr=16),
    "LPDDR5": dict(org="LPDDR5_16Gb_x16", timing="LPDDR5_6400",
                   extra={"channel_width": 32}, mshr=16),
    "LPDDR6": dict(org="LPDDR6_16Gb_x12", timing="LPDDR6_10667_BL24",
                   extra={}, mshr=16),
    "HBM4": dict(org="HBM4_32Gb_8Hi", timing="HBM4_8000Mbps",
                 extra={"channel_width": 64}, mshr=16),
}

# ── SimpleO3 workloads ───────────────────────────────────────────────────
CHRONUS = ["429.mcf", "519.lbm", "549.fotonik3d", "459.GemsFDTD", "450.soplex",
           "433.milc", "462.libquantum", "483.xalancbmk", "401.bzip2", "435.gromacs"]
DPC4 = ["605.mcf_s-1554B", "619.lbm_s-2676B", "649.fotonik3d_s-1176B",
        "654.roms_s-1021B", "603.bwaves_s-1080B", "623.xalancbmk_s-165B"]
MIXES = {
    "Mix1": ["519.lbm", "450.soplex", "434.zeusmp", "483.xalancbmk"],
    "Mix2": ["459.GemsFDTD", "434.zeusmp", "482.sphinx3", "549.fotonik3d"],
    "MixS17a": ["605.mcf_s-1554B", "619.lbm_s-2676B", "649.fotonik3d_s-1176B",
                "623.xalancbmk_s-165B"],
}
INSTS_SINGLE = 20_000_000
INSTS_MIX = 10_000_000

# ── gem5 ─────────────────────────────────────────────────────────────────
GEM5_BIN = os.environ.get("GEM5_BIN", "/home/dev/gem5-src/build/X86/gem5.opt")
GEM5_BENCH_DIR = pathlib.Path(os.environ.get("GEM5_BENCH_DIR", TRACES / "gem5bench/bin"))
GEM5_BENCH = {
    "stream": "", "gups": "", "ptrchase": "", "matmul": "",
    "bfs": "-g 16 -n 1", "pr": "-g 15 -i 5 -n 1",
    "pb_atax": "", "pb_mvt": "", "pb_bicg": "", "pb_gemm": "",
    "pb_jacobi2d": "", "pb_heat3d": "",
}

# ── ChampSim ─────────────────────────────────────────────────────────────
CHAMPSIM_DIR = pathlib.Path(os.environ.get("CHAMPSIM_DIR", "/home/dev/champsim"))
CHAMPSIM_BIN = pathlib.Path(os.environ.get("CHAMPSIM_BIN", CHAMPSIM_DIR / "bin" / "champsim"))
CHAMPSIM_TRACES = pathlib.Path(os.environ.get("CHAMPSIM_TRACES", TRACES / "dpc4"))
CHAMPSIM_WORKLOADS = DPC4  # same six SPEC17 traces, .champsimtrace.xz

# Bump these when the on-disk contracts change.  The request metric version is
# deliberately separate from matchlib's matcher version: changing a formula or
# normalization must invalidate request.json even when request pairing does not.
RUN_MANIFEST_SCHEMA_VERSION = 5
REQUEST_METRIC_SCHEMA_VERSION = 6


def trace_path(wl):
    p = TRACES / "chronus" / "cputraces" / wl
    return str(p) if p.exists() else str(TRACES / "dpc4" / "converted" / f"{wl}.trace")


def simpleo3_trace_inputs(workloads):
    """Hash each unique external SimpleO3 trace once for a parent run."""
    unique = sorted({trace for wl in workloads for trace in MIXES.get(wl, [wl])})
    fingerprints = {trace: file_provenance(trace_path(trace)) for trace in unique}
    return {
        wl: [fingerprints[trace] for trace in MIXES.get(wl, [wl])]
        for wl in workloads
    }


def validate_simpleo3_trace_inputs(trace_inputs):
    """Fail if a parent-captured external trace identity no longer matches."""
    unique = {}
    for fingerprints in trace_inputs.values():
        for fingerprint in fingerprints:
            unique[fingerprint["path"]] = fingerprint
    changed = [
        path for path, fingerprint in unique.items()
        if not file_provenance_matches(path, fingerprint)
    ]
    if changed:
        raise RuntimeError(
            "external SimpleO3 trace inputs changed during the run: "
            + ", ".join(changed)
        )


def file_provenance(path):
    """Return a content identity for an external input or raw artifact."""
    path = pathlib.Path(path)
    stat_before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (
            stat_before.st_size, stat_before.st_mtime_ns):
        raise RuntimeError(f"file changed while hashing: {path}")
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def file_provenance_matches(path, recorded):
    """Check the durable parts of a recorded file identity.

    mtime is retained for audit/debugging, but it is not an identity field: a
    byte-identical restored artifact remains valid.
    """
    if not isinstance(recorded, dict):
        return False
    current = file_provenance(path)
    return all(current[key] == recorded.get(key)
               for key in ("path", "size", "sha256"))


def atomic_write_json(path, payload):
    """Durably publish JSON with a same-directory atomic rename."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            json.dump(payload, stream, indent=1)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def stage_run_directory(outdir, *metadata_names):
    """Invalidate derived metadata and allocate a same-filesystem staging dir.

    Existing raw traces are intentionally left in place until a replacement
    run has completed, so a failed forced rerun cannot destroy the last raw.
    The missing manifest makes that old raw unambiguously non-cacheable.
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in metadata_names:
        (outdir / name).unlink(missing_ok=True)
    return pathlib.Path(tempfile.mkdtemp(prefix=".incomplete-", dir=outdir))


def publish_trace_and_manifest(staged_trace, trace_path, manifest_path, payload):
    """Publish a completed raw trace, then its checksum-bearing manifest."""
    staged_trace = pathlib.Path(staged_trace)
    trace_path = pathlib.Path(trace_path)
    manifest_path = pathlib.Path(manifest_path)
    if not staged_trace.is_file() or staged_trace.stat().st_size == 0:
        raise RuntimeError(f"simulation produced no non-empty raw trace: {staged_trace}")
    manifest_path.unlink(missing_ok=True)
    os.replace(staged_trace, trace_path)
    raw_trace = file_provenance(trace_path)
    atomic_write_json(manifest_path, {**payload, "raw_trace": raw_trace})
    return raw_trace


def git_rev():
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def git_dirty():
    out = subprocess.run([
        "git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all",
    ],
                         capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def git_dirty_sha256():
    """Hash tracked diffs plus every non-ignored untracked file exactly."""
    status = subprocess.run([
        "git", "-C", str(REPO), "status", "--porcelain=v1", "-z",
        "--untracked-files=all",
    ], check=True, capture_output=True).stdout
    diff = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--binary", "HEAD", "--"],
        check=True, capture_output=True,
    ).stdout
    digest = hashlib.sha256()
    digest.update(b"status\0")
    digest.update(status)
    digest.update(b"diff\0")
    digest.update(diff)
    for record in status.split(b"\0"):
        if not record.startswith(b"?? "):
            continue
        relative = os.fsdecode(record[3:])
        path = REPO / relative
        digest.update(b"untracked\0")
        digest.update(os.fsencode(relative))
        digest.update(b"\0")
        stat_before = path.lstat()
        if path.is_symlink():
            digest.update(os.fsencode(os.readlink(path)))
        elif path.is_file():
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        else:
            raise RuntimeError(f"untracked provenance path is not a file: {path}")
        stat_after = path.lstat()
        if (stat_before.st_mode, stat_before.st_size, stat_before.st_mtime_ns) != (
                stat_after.st_mode, stat_after.st_size, stat_after.st_mtime_ns):
            raise RuntimeError(f"untracked provenance file changed while hashing: {path}")
    status_after = subprocess.run([
        "git", "-C", str(REPO), "status", "--porcelain=v1", "-z",
        "--untracked-files=all",
    ], check=True, capture_output=True).stdout
    diff_after = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--binary", "HEAD", "--"],
        check=True, capture_output=True,
    ).stdout
    if status_after != status or diff_after != diff:
        raise RuntimeError("repository changed while evaluation provenance was being captured")
    return digest.hexdigest()


def mapped_native_library_path(maps_path="/proc/self/maps",
                               library_name="libramulator.so"):
    """Resolve the unique, non-deleted native library mapped in this process."""
    try:
        lines = pathlib.Path(maps_path).read_text().splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot inspect mapped native libraries: {exc}") from exc
    paths = set()
    deleted = []
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        raw_path = fields[5]
        is_deleted = raw_path.endswith(" (deleted)")
        candidate = raw_path.removesuffix(" (deleted)")
        if pathlib.Path(candidate).name != library_name:
            continue
        resolved = str(pathlib.Path(candidate).resolve())
        if is_deleted:
            deleted.append(resolved)
        else:
            paths.add(resolved)
    if deleted:
        raise RuntimeError(
            f"mapped {library_name} has been deleted/replaced: {sorted(set(deleted))}"
        )
    if len(paths) != 1:
        raise RuntimeError(
            f"expected exactly one mapped {library_name}, found {sorted(paths)}"
        )
    return pathlib.Path(next(iter(paths)))


def native_library_provenance():
    """Fingerprint the exact native library mapped by the Python extension."""
    # Loading the extension first makes /proc/self/maps authoritative. RUNPATH
    # alone is insufficient because LD_LIBRARY_PATH can override it.
    importlib.import_module("ramulator._ramulator")
    expected = (REPO / "libramulator.so").resolve()
    if not expected.is_file():
        raise RuntimeError(
            f"missing native Ramulator library {expected}; build it before running evaluations"
        )
    mapped = mapped_native_library_path()
    if mapped != expected:
        raise RuntimeError(
            f"Python mapped {mapped}, but evaluation expects {expected}; "
            "check LD_LIBRARY_PATH and restart the process"
        )
    return file_provenance(mapped)


def python_extension_provenance():
    """Fingerprint the exact CPython extension loaded by this evaluator.

    ``libramulator.so`` alone is insufficient provenance: an older pybind
    extension can keep a stale ABI or binding implementation while resolving
    against a newly-built shared library.  Importing the module first makes
    this record describe the artifact Python actually loaded, rather than a
    guessed build-tree path.
    """
    module = importlib.import_module("ramulator._ramulator")
    origin = getattr(module, "__file__", None)
    if not origin:
        raise RuntimeError("loaded ramulator._ramulator has no filesystem origin")
    path = pathlib.Path(origin).resolve()
    if (not path.name.startswith("_ramulator") or
            not any(path.name.endswith(suffix)
                    for suffix in importlib.machinery.EXTENSION_SUFFIXES)):
        raise RuntimeError(
            "ramulator._ramulator did not resolve to a native Python extension: "
            f"{path}"
        )
    return file_provenance(path)


def optimized_build_provenance(build_dir=OPT_BUILD):
    """Fail closed unless evaluation uses the configured `-O3` build.

    Ramulator's build places the shared library in the repository root, so its
    path alone does not distinguish Debug and Release builds. The evaluator
    records the CMake cache and actual simulator target flags and requires
    both Release mode and an explicit `-O3` flag.
    """
    build_dir = pathlib.Path(build_dir).resolve()
    cache = build_dir / "CMakeCache.txt"
    flags = build_dir / "CMakeFiles" / "ramulator.dir" / "flags.make"
    missing = [str(path) for path in (cache, flags) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "missing optimized-build evidence: " + ", ".join(missing)
        )
    cache_text = cache.read_text(errors="replace")
    flags_text = flags.read_text(errors="replace")
    if "CMAKE_BUILD_TYPE:STRING=Release" not in cache_text:
        raise RuntimeError(f"evaluation build is not Release: {build_dir}")
    cxx_flag_lines = [
        line for line in flags_text.splitlines() if line.startswith("CXX_FLAGS")
    ]
    if not cxx_flag_lines or not any(
            "-O3" in line.split() for line in cxx_flag_lines):
        raise RuntimeError(
            f"evaluation simulator target is not compiled with -O3: {flags}"
        )
    return {
        "build_dir": str(build_dir),
        "build_type": "Release",
        "optimization": "-O3",
        "cmake_cache": file_provenance(cache),
        "target_flags": file_provenance(flags),
    }


def provenance_identity(provenance):
    """Stable identity for the source tree and loaded native artifacts."""
    identity = {
        "rev": provenance["rev"],
        "dirty_sha256": provenance["dirty_sha256"],
        "native_library_sha256": provenance["native_library"]["sha256"],
        "python_extension_sha256": provenance["python_extension"]["sha256"],
        "build_flags_sha256": provenance["optimized_build"]["target_flags"]["sha256"],
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def runtime_provenance():
    """Capture source and native-artifact identity for a model evaluation."""
    provenance = {
        "rev": git_rev(),
        "dirty": git_dirty(),
        "dirty_sha256": git_dirty_sha256(),
        "native_library": native_library_provenance(),
        "python_extension": python_extension_provenance(),
        "optimized_build": optimized_build_provenance(),
    }
    provenance["identity"] = provenance_identity(provenance)
    return provenance


def validate_runtime_provenance(expected):
    """Fail if source or native code changed after a parent captured identity."""
    current = runtime_provenance()
    expected_identity = expected.get("identity", provenance_identity(expected))
    if current["identity"] != expected_identity:
        raise RuntimeError(
            "evaluation source/native provenance changed during the run; "
            "discard the incomplete batch and rerun"
        )
    return current
