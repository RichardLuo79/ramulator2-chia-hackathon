"""Opt-in real -O3/full-window recovery test; no LLM calls or accuracy claims.

CHIA_FULL_WINDOW_TEST=1 python -m pytest -q tests/unit_tests/test_chia_unattended_integration.py
Requires the usual optimized build and licensed trace installation. Artifacts
are temporary test outputs, never feedback for an optimization agent.
"""
import os
import pathlib

import pytest

from tools.chia_loop import real_core as P, real_eval as E, recovery as R


@pytest.mark.skipif(os.environ.get("CHIA_FULL_WINDOW_TEST") != "1", reason="opt-in real full-window evaluation")
def test_optimized_build_and_full_window_trace_cache_survive_restart(tmp_path, monkeypatch):
    from google import genai
    monkeypatch.setattr(genai, "Client", lambda *a, **kw: pytest.fail("integration test must never call an LLM"))
    root = tmp_path / "full_window_recovery"
    root.mkdir()
    seed = (E.REPO / P.MUTABLE).read_text()
    with R.cpu_lease(E.REPO / "eval_out/chia/.cpu_leases", 3):
        E.prepare_runtime(root)
        plugin = E.compile_candidate(root, seed, root / "seed", resume=True)
        before_build = (root / "seed/build.json").read_bytes()
        assert E.compile_candidate(root, seed, root / "seed", resume=True) == plugin
        assert (root / "seed/build.json").read_bytes() == before_build
        workload = E.TRAIN[0]
        for model, label, library in (("oracle", "oracle", None), ("candidate", "seed", plugin)):
            first = E.run_one(root, workload, model, label, library, split="training", resume=True)
            assert first["insts_per_core"] == 20_000_000 and first["optimization"] == "-O3"
            directory = root / "training/simpleo3/DDR5" / workload / label
            assert not (directory / "trace.csv.ch0").exists()
            before = (directory / "manifest.json").read_bytes()
            with monkeypatch.context() as local:
                local.setattr(E, "sandbox_command", lambda *a, **kw: pytest.fail("completed simulation reran"))
                cached = E.run_one(root, workload, model, label, library, split="training", resume=True)
            assert cached == first and (directory / "manifest.json").read_bytes() == before
        assert E.verify_run_archives(root) == 4
        with pytest.raises(R.OperationalPause, match="identity"):
            E.compile_candidate(root, seed + "\n", root / "seed", resume=True)
