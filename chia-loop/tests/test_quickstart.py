"""Public preparation and preflight fixtures; no provider requests."""

from contextlib import contextmanager
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import threading

import pytest

from ramulator_chia import commands, configuration, input_data, preflight
from ramulator_chia.layout import ROOT


@contextmanager
def range_server(payload, mode="normal"):
    state = {"mode": mode, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            assert self.headers["If-Match"] == '"fixture-etag"'
            start, end = map(
                int, self.headers["Range"].removeprefix("bytes=").split("-")
            )
            state["requests"].append((start, end))
            current = state["mode"]
            self.send_response(200 if current == "ignore-range" else 206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.send_header(
                "ETag", '"changed"' if current == "changed" else '"fixture-etag"'
            )
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            data = payload[start : end + 1]
            if current == "interrupt":
                state["mode"] = "normal"
                data = data[: max(1, len(data) // 2)]
                self.close_connection = True
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/", state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def source(payload, decoded=None):
    return {
        "path": "fixture.gz",
        "etag": "fixture-etag",
        "size_bytes": len(payload),
        "upstream_range_bytes": len(payload),
        "prefix_instructions": 2,
        "decoded_sha256": hashlib.sha256(decoded or payload).hexdigest(),
    }


def test_interrupted_range_resumes(tmp_path):
    payload = bytes(range(128))
    with range_server(payload, "interrupt") as (url, state):
        partial = tmp_path / "source.partial"
        with pytest.raises(RuntimeError, match="resume"):
            input_data.download_range(url, source(payload), partial, reserve=0)
        assert partial.stat().st_size == 64
        input_data.download_range(url, source(payload), partial, reserve=0)
        assert partial.read_bytes() == payload
        assert state["requests"] == [(0, 127), (64, 127)]


@pytest.mark.parametrize("mode", ["changed", "ignore-range"])
def test_untrusted_response_is_not_appended(tmp_path, mode):
    with range_server(b"abc", mode) as (url, state):
        partial = tmp_path / "source.partial"
        with pytest.raises(ValueError, match="changed|ignored"):
            input_data.download_range(url, source(b"abc"), partial, reserve=0)
        assert not partial.exists()


def test_prefix_and_repeated_preparation(tmp_path):
    decoded = bytes(range(128))
    payload = gzip.compress(decoded + b"additional records", mtime=1)
    metadata = source(payload, decoded)
    with range_server(payload) as (url, state):
        first = input_data.prepare_trace(
            tmp_path, "traces/fixture.gz", metadata, url, reserve=0
        )
        second = input_data.prepare_trace(
            tmp_path, "traces/fixture.gz", metadata, url, reserve=0
        )
        assert first == second
        assert first["decoded_bytes"] == 128
        assert gzip.decompress((tmp_path / "traces/fixture.gz").read_bytes()) == decoded
        assert len(state["requests"]) == 1


@pytest.mark.parametrize("error", ["short", "corrupt", "hash"])
def test_bad_prefix_never_publishes(tmp_path, error):
    decoded = bytes(range(128))
    payload = gzip.compress(decoded[:64] if error == "short" else decoded)
    if error == "corrupt":
        payload = payload[:10] + b"invalid deflate"
    metadata = source(payload, b"wrong" if error == "hash" else decoded)
    with range_server(payload) as (url, state):
        with pytest.raises((ValueError, OSError, EOFError)):
            input_data.prepare_trace(
                tmp_path, "traces/fixture.gz", metadata, url, reserve=0
            )
    assert not (tmp_path / "traces/fixture.gz").exists()


def test_space_and_symlinks_fail_before_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        input_data.shutil, "disk_usage", lambda _: type("Disk", (), {"free": 100})()
    )
    with pytest.raises(RuntimeError, match="space"):
        input_data.space_check(tmp_path, 100, 32)
    (tmp_path / "traces").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        input_data.safe_path(tmp_path, "traces/file.gz")


@pytest.mark.parametrize(
    "name,rounds", [("astra", 15), ("deepseek", 15), ("gemini", 15), ("opus", 10)]
)
def test_four_templates_and_compaction(tmp_path, monkeypatch, name, rounds):
    monkeypatch.setenv("CHIA_VERTEX_PROJECT", "reader-project")
    cfg = configuration.load_config(
        ROOT / f"chia-loop/configs/{name}.json", tmp_path / "data", tmp_path / "work"
    )
    assert cfg.run.maximum_iterations == rounds
    assert cfg.run.resources.simulation_timeout_seconds is None
    assert cfg.run.resources.diagnostic_timeout_seconds == 1800
    assert cfg.run.model_timeout_seconds == 10800
    assert cfg.experiment.promotion_policy == "llm_review"
    assert cfg.experiment.prompt_sha256
    policy = configuration.compaction_policy(cfg)
    assert (policy is not None) == (name in {"gemini", "deepseek"})
    if name == "gemini":
        assert cfg.backend.project == "reader-project"


def test_gemini_requires_explicit_project(tmp_path, monkeypatch):
    monkeypatch.delenv("CHIA_VERTEX_PROJECT", raising=False)
    with pytest.raises(ValueError, match="VERTEX_PROJECT"):
        configuration.load_config(
            ROOT / "chia-loop/configs/gemini.json", tmp_path, tmp_path
        )


def test_check_does_not_create_campaign_or_invoke_launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preflight, "check", lambda *a, **k: {"passed": True, "inference_calls": 0}
    )
    monkeypatch.setattr(
        commands.runpy, "run_module", lambda *a, **k: pytest.fail("launch forbidden")
    )
    output = tmp_path / "not-created"
    commands.run_campaign(
        [
            "--check",
            "--config",
            str(ROOT / "chia-loop/configs/astra.json"),
            "--output",
            str(output),
            "--tariff",
            str(ROOT / "chia-loop/configs/tariffs/astra.json"),
        ]
    )
    assert not output.exists()


def test_real_isolation_probe():
    assert "passed" in preflight.isolation_check()


def test_oracle_comparison_uses_the_existing_verified_matcher(tmp_path, monkeypatch):
    from ramulator_chia.framework import evaluation, inputs
    from test_chia_framework_transfer_requests import measurement

    calls = []
    monkeypatch.setattr(
        inputs,
        "_staged_champsim",
        lambda *a: ({"validation": [object()]}, {"champsim:1": object()}),
    )

    def measured(runtime, case, model, output, **kwargs):
        calls.append(model)
        rows = ["0,10,0,0,64,0,0,0"] if model == "oracle" else ["0,11,0,0,64,0,0,0"]
        fixture, digest = measurement(tmp_path / "fixtures", model, rows)
        shutil.copytree(fixture, output)
        return json.loads((output / "measurement.json").read_text())

    monkeypatch.setattr(evaluation, "measure_native", measured)
    output = tmp_path / "comparison"
    commands.evaluate(
        [
            "--case",
            "validation-c1-bwaves",
            "--model",
            "fixedlat",
            "--output",
            str(output),
            "--with-oracle",
        ]
    )
    result = json.loads((output / "comparison.json").read_text())
    assert calls == ["fixedlat", "oracle"]
    assert result["cycles"]["mean_abs_per_core_pct"] == 10
    assert result["request"]["mae"] == 0.1


def test_preparation_has_a_single_owner(tmp_path):
    with input_data.preparation_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another input preparation"):
            with input_data.preparation_lock(tmp_path):
                pytest.fail("second owner acquired the lock")


def test_source_inventory_matches_all_frozen_inputs():
    sources = json.loads((ROOT / "results/manifests/input-sources.json").read_text())[
        "traces"
    ]
    frozen = json.loads((ROOT / "results/manifests/inputs.json").read_text())["files"]
    traces = {r["path"]: r for r in frozen if r["kind"] == "trace"}
    assert len(sources) == len(traces) == 52
    assert sum(r["kind"] == "placement" for r in frozen) == 100
    for name, row in sources.items():
        assert row["decoded_sha256"] == traces[name]["decoded_sha256"]
        assert row["prefix_instructions"] == 23_000_000
        assert 0 < row["upstream_range_bytes"] <= row["size_bytes"]


def test_prepared_identity_rejects_changed_bytes_or_decoded_identity(tmp_path):
    name = "traces/fixture.gz"
    path = tmp_path / name
    path.parent.mkdir()
    path.write_bytes(gzip.compress(b"fixture"))
    original = "a" * 64
    logical = hashlib.sha256(b"fixture").hexdigest()
    item = {"path": str(path), "sha256": original, "decoded_sha256": logical}
    config = {
        "experiment": {
            "evaluation": {"champsim": {"traces": {"fixture": item}, "cases": {}}}
        }
    }
    row = {
        "sha256": input_data.sha(path),
        "decoded_sha256": logical,
        "reference_sha256": original,
    }
    receipt = {
        "schema_version": 1,
        "source_manifest_sha256": input_data.sha(
            ROOT / "results/manifests/input-sources.json"
        ),
        "files": {name: row},
    }
    input_data.write_json(tmp_path / "prepared-inputs.json", receipt)
    result = configuration.bind_prepared(config, tmp_path)
    assert (
        result["experiment"]["evaluation"]["champsim"]["traces"]["fixture"]["sha256"]
        == row["sha256"]
    )
    item["decoded_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="differs"):
        configuration.bind_prepared(config, tmp_path)
    item["decoded_sha256"] = logical
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        configuration.bind_prepared(config, tmp_path)
