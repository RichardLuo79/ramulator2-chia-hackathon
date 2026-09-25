"""Behavioral tests for ChampSim smoke trace generator and Dockerfile structure.

Gap 1: Trace generator correctness — verify that generate() produces
        valid 64-byte-per-instruction traces.
Gap 2: Committed smoke.champsimtrace.gz has correct decompressed size (640000 bytes).
Gap 3: ChampSimDockerfile contains required strings (pinned SHA, vcpkg, next_line)
        and must NOT contain --depth 1.
"""
from __future__ import annotations

import gzip
import struct
import sys
from pathlib import Path

import pytest

# This file lives at chia/simulators/tests/; project root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TESTS_DIR = Path(__file__).resolve().parent

TRACE_GENERATOR = TESTS_DIR / "traces" / "generate_smoke_trace.py"
SMOKE_TRACE = TESTS_DIR / "traces" / "smoke.champsimtrace.gz"
CHAMPSIM_DOCKERFILE = PROJECT_ROOT / "dockerfiles" / "ChampSimDockerfile"


# ---------------------------------------------------------------------------
# Gap 1: trace generator correctness
# ---------------------------------------------------------------------------

def test_trace_generator_produces_64_bytes_per_instruction(tmp_path):
    """generate() output size must equal num_instructions * 64 bytes."""
    # Import the generator module dynamically without adding it to sys.path globally
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_smoke_trace", TRACE_GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    num = 50
    out = tmp_path / "out.champsimtrace.gz"
    mod.generate(str(out), num_instructions=num)

    with gzip.open(str(out), "rb") as f:
        data = f.read()

    assert len(data) == num * 64, (
        f"Expected {num * 64} bytes for {num} instructions, got {len(data)}"
    )


def test_instr_fmt_constant_is_64_bytes():
    """INSTR_FMT struct format must pack to exactly 64 bytes."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_smoke_trace", TRACE_GENERATOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert struct.calcsize(mod.INSTR_FMT) == 64, (
        f"INSTR_FMT calcsize is {struct.calcsize(mod.INSTR_FMT)}, expected 64"
    )


# ---------------------------------------------------------------------------
# Gap 2: committed smoke trace has correct decompressed size
# ---------------------------------------------------------------------------

def test_smoke_trace_decompressed_size_is_640000_bytes():
    """smoke.champsimtrace.gz must decompress to exactly 640000 bytes (10000 * 64)."""
    assert SMOKE_TRACE.exists(), f"Smoke trace not found: {SMOKE_TRACE}"

    with gzip.open(str(SMOKE_TRACE), "rb") as f:
        data = f.read()

    expected = 10000 * 64
    assert len(data) == expected, (
        f"Expected {expected} bytes (10000 instructions * 64 bytes), got {len(data)}"
    )


# ---------------------------------------------------------------------------
# Gap 3: Dockerfile structure
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dockerfile_text():
    assert CHAMPSIM_DOCKERFILE.exists(), f"Dockerfile not found: {CHAMPSIM_DOCKERFILE}"
    return CHAMPSIM_DOCKERFILE.read_text()


def test_dockerfile_has_base_image(dockerfile_text):
    assert "FROM rayproject/ray:2.54.0-cpu" in dockerfile_text, (
        "Dockerfile missing base image FROM rayproject/ray:2.54.0-cpu (D-02)"
    )


def test_dockerfile_has_pinned_ref(dockerfile_text):
    assert "CHAMPSIM_REF=" in dockerfile_text, (
        "Dockerfile missing pinned CHAMPSIM_REF ARG"
    )


def test_dockerfile_has_vcpkg_bootstrap(dockerfile_text):
    assert "vcpkg/bootstrap-vcpkg.sh" in dockerfile_text, (
        "Dockerfile missing vcpkg/bootstrap-vcpkg.sh"
    )


def test_dockerfile_has_vcpkg_install(dockerfile_text):
    assert "vcpkg/vcpkg install" in dockerfile_text, (
        "Dockerfile missing vcpkg/vcpkg install"
    )


def test_dockerfile_has_next_line_prefetcher(dockerfile_text):
    assert "next_line" in dockerfile_text, (
        "Dockerfile missing next_line prefetcher config (D-04)"
    )


def test_dockerfile_has_no_depth_1_flag(dockerfile_text):
    assert "--depth 1" not in dockerfile_text, (
        "Dockerfile must NOT use --depth 1 (D-09 requires full clone for git diff support)"
    )


def test_dockerfile_copies_smoke_trace(dockerfile_text):
    assert "smoke.champsimtrace.gz" in dockerfile_text, (
        "Dockerfile missing COPY of smoke.champsimtrace.gz (D-06)"
    )


def test_dockerfile_installs_chia_last(dockerfile_text):
    """chia pip install must appear after all ChampSim build steps."""
    chia_copy_idx = dockerfile_text.rfind("COPY --chown=ray:ray . /tmp/chia")
    vcpkg_idx = dockerfile_text.find("vcpkg/vcpkg install")
    assert chia_copy_idx > vcpkg_idx, (
        "COPY . /tmp/chia must appear after vcpkg install for optimal layer caching (D-03)"
    )
