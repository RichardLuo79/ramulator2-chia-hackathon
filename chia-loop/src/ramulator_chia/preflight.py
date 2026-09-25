"""Local campaign checks. Never authenticate, refresh a token, or call a model."""

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

from .input_data import sha
from .layout import ROOT


def credential_file(path, output):
    path = Path(path).expanduser()
    if path.is_symlink():
        raise ValueError("credential file must not be a symlink")
    path = path.resolve(strict=True)
    if path.is_relative_to(ROOT) or path.is_relative_to(output):
        raise ValueError("keep credentials outside the artifact and campaign output")
    info = path.stat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("credential file must be owner-only (chmod 600)")
    return path


def provider_check(backend, output):
    kind = backend.kind
    if kind in {"codex_cli", "claude_cli"}:
        from .framework.transport import CLI

        profile = CLI[kind]
        binary = shutil.which(profile["binary"])
        if not binary:
            raise ValueError("install the provider CLI and set its CHIA_*_BIN path")
        binary = Path(binary).resolve(strict=True)
        # The protected profile exposes the native binary, not an npm tree.
        with binary.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                raise ValueError(
                    "CHIA_*_BIN must name the native ELF executable, not a shell/npm wrapper"
                )
        for name in profile["companions"]:
            if not (binary.parent / name).is_file():
                raise ValueError("provider installation lacks companion: " + name)
        if kind == "codex_cli":
            from .codex_cli.auth import read_tokens

            path = credential_file(
                os.environ.get("CHIA_CODEX_AUTH", "~/.codex/auth.json"), output
            )
            read_tokens(path)
        else:
            from .claude_cli.auth import read_setup_token

            path = credential_file(
                os.environ.get("CHIA_CLAUDE_AUTH", "~/.claude/chia-oauth-token"), output
            )
            read_setup_token(path)
        return {
            "kind": kind,
            "binary_sha256": sha(binary),
            "credential": "present; not authenticated",
        }
    if kind == "deepseek_api":
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key or any(c.isspace() for c in key):
            raise ValueError("set DEEPSEEK_API_KEY in the trusted launch environment")
    elif kind == "vertex_gemini":
        path = credential_file(
            os.environ.get(
                "GOOGLE_APPLICATION_CREDENTIALS",
                "~/.config/gcloud/application_default_credentials.json",
            ),
            output,
        )
        value = json.loads(path.read_text())
        if value.get("type") not in {
            "authorized_user",
            "service_account",
            "external_account",
        }:
            raise ValueError("unsupported local ADC credential file")
    else:
        raise ValueError(
            "public quickstart supports codex_cli, claude_cli, deepseek_api, vertex_gemini"
        )
    return {"kind": kind, "credential": "present; not authenticated"}


def isolation_check():
    """Exercise the real fail-closed boundary in a disposable child process."""
    with tempfile.TemporaryDirectory(prefix="chia-preflight-") as temporary:
        scratch = Path(temporary)
        allowed = scratch / "allowed"
        allowed.mkdir()
        secret = scratch / "denied"
        secret.write_text("preflight sentinel")
        code = """
import pathlib, socket, sys
from ramulator_chia.sandbox import restrict
allowed, denied = map(pathlib.Path, sys.argv[1:])
restrict(read=[str(allowed)], write=[str(allowed)], isolate_processes=True)
(allowed/'check').write_text('ok')
assert (allowed/'check').read_text() == 'ok'
try:
    denied.read_bytes()
except PermissionError:
    pass
else:
    raise RuntimeError('private-file boundary failed')
try:
    socket.socket()
except PermissionError:
    pass
else:
    raise RuntimeError('network boundary failed')
"""
        result = subprocess.run(
            [sys.executable, "-c", code, str(allowed), str(secret)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            raise RuntimeError(
                "Landlock ABI 6/seccomp isolation probe failed; use a compatible kernel"
            )
    return "passed: allowed workspace, denied private file and network"


def runtime_check(runtime):
    runtime = Path(runtime).absolute()
    manifest = json.loads((runtime / "runtime_manifest.json").read_text())
    for relative, digest in manifest["source_inventory"].items():
        if sha(runtime / "runtime-source" / relative) != digest:
            raise ValueError("staged runtime source changed: " + relative)
    for relative, field in {
        "runtime/isolated_sim": "executable_sha256",
        "runtime/libramulator.so": "library_sha256",
        "runtime/candidate.so": "seed_plugin_sha256",
        "runtime-build/compile_commands.json": "compile_commands_sha256",
    }.items():
        if sha(runtime / relative) != manifest[field]:
            raise ValueError("runtime changed: " + relative)
    for relative, digest in manifest["export_hashes"].items():
        if sha(runtime / "export" / relative) != digest:
            raise ValueError("runtime API export changed")
    compiler = manifest["compiler"]
    if sha(compiler["path"]) != compiler["sha256"]:
        raise ValueError(
            "compiler changed since runtime build; rebuild in a new directory"
        )
    if manifest["optimization"] != "-O3":
        raise ValueError("optimized runtime required")
    return sha(runtime / "runtime_manifest.json")


def check(configuration, runtime, output, tariff, *, workers=None, compaction=None):
    from .framework.config import ExecutionOverrides, preserved_configuration
    from .framework.model_sessions import upstream_identity
    from .framework.usage import Tariff

    output = Path(output).absolute()
    report = {
        "inference_calls": 0,
        "campaign_created": False,
        "checks": {},
        "errors": [],
    }

    def inspect(name, action):
        try:
            report["checks"][name] = action()
        except (
            OSError,
            ValueError,
            RuntimeError,
            ImportError,
            subprocess.SubprocessError,
        ) as exc:
            # Provider setup failures must not disclose a parsed credential body.
            detail = type(exc).__name__ if name == "provider" else str(exc)
            report["errors"].append({"check": name, "error": detail})

    inspect("adapter_sources", upstream_identity)
    inspect("runtime", lambda: runtime_check(runtime))
    inspect(
        "tariff",
        lambda: Tariff.model_validate_json(Path(tariff).read_text()).model_dump(
            mode="json"
        ),
    )

    def inputs():
        cohort = configuration.experiment.evaluation.champsim
        items = list(cohort.traces.values()) + [
            case.placement for case in cohort.cases.values()
        ]
        for item in items:
            path = Path(item.path)
            if path.is_symlink() or sha(path) != item.sha256:
                raise ValueError("input changed: " + path.name)
        for build in cohort.builds.values():
            if not os.access(build.binary, os.X_OK) or not Path(build.source).is_dir():
                raise ValueError("missing ChampSim build")
        return {
            "traces": len(cohort.traces),
            "placements": len(cohort.cases),
            "frontends": sorted(cohort.builds),
        }

    inspect("inputs", inputs)

    def prompts():
        from .framework.review import prompt_inventory

        expected = configuration.experiment.prompt_sha256
        if not expected or prompt_inventory() != expected:
            raise ValueError("staged prompts differ from the configuration pins")
        return expected

    inspect("prompts", prompts)

    def host():
        effective = ExecutionOverrides(evaluation_workers=workers).workers(
            configuration
        )
        if effective > len(os.sched_getaffinity(0)):
            raise ValueError("evaluation workers exceed the allowed CPU affinity")
        parent = output
        while not parent.exists():
            parent = parent.parent
        if not parent.is_dir() or not os.access(parent, os.W_OK):
            raise ValueError("campaign output parent is not writable")
        if any(p.is_symlink() for p in (output, *output.parents)):
            raise ValueError("campaign output must not use symlinks")
        saved = output / "config.json"
        preserved_configuration(
            configuration, json.loads(saved.read_text()) if saved.exists() else None
        )
        return {
            "workers": effective,
            "available_cpus": len(os.sched_getaffinity(0)),
            "output": "existing campaign" if saved.exists() else "new directory",
        }

    inspect("host", host)
    inspect("provider", lambda: provider_check(configuration.backend, output))
    if configuration.backend.kind in {"deepseek_api", "vertex_gemini"}:

        def context():
            if compaction is None:
                raise ValueError("API backend requires its explicit compaction policy")
            script = "from importlib.metadata import version; from google.adk.apps.llm_event_summarizer import LlmEventSummarizer; assert version('google-adk') == '2.3.0'"
            subprocess.run(
                [compaction.adk_python, "-c", script],
                check=True,
                capture_output=True,
                timeout=30,
            )
            return compaction.model_dump(mode="json")

        inspect("compaction", context)
    inspect("isolation", isolation_check)
    report["passed"] = not report["errors"]
    return report
