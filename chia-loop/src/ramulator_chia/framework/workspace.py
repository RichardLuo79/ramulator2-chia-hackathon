"""The shared agent-facing file and training-tool surface.

Only these methods become MCP tools. The native process boundary additionally
mounts the listed directories with the same permissions; it never mounts the
campaign root. This module is also used by the offline scripted-agent checks.
"""

from __future__ import annotations

import fcntl
import gzip
import json
import lzma
import os
import struct
from functools import wraps
from itertools import islice
from pathlib import Path

from pydantic_core import to_jsonable_python

from ramulator_chia.eval import artifacts

from .archive import safe_name
from .build import MODEL_API_HEADER
from .diagnostic_cases import SyntheticRequest
from .dram import COMPARISONS, MODEL_FILES
from .identity import canonical_json, file_sha256
from .snapshots import (
    materialize,
    publish_bytes,
    read_file,
    replace_file,
    snapshot,
    verify_snapshot,
)


def recorded(function):
    @wraps(function)
    def invoke(self, *args, **kwargs):
        event = {"tool": function.__name__, "args": args, "kwargs": kwargs, "phase": self.phase}
        try:
            result = function(self, *args, **kwargs)
            event["result"] = result
            return result
        except Exception as exc:
            event["error"] = {"type": type(exc).__name__, "message": str(exc)}
            failure = getattr(exc, "receipt", {}).get("diagnostic_failure")
            if failure:
                event["diagnostic_failure"] = failure
            raise
        finally:
            self.record(event)

    return invoke


class Workspace:
    def __init__(self, research, iteration: int, role: str, candidate: dict, history: list[dict], *, review_context=None):
        if role not in ("proposer", "reviewer") or (role == "reviewer" and history):
            raise ValueError("reviewers receive a draft and rules, not proposer history")
        self.research, self.iteration, self.role = research, iteration, role
        self.phase = "prepared"
        self.root = research.root / "agents" / str(iteration) / role
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = research.root / "events" / f"iteration-{iteration}-{role}.jsonl.gz"
        self.events.parent.mkdir(parents=True, exist_ok=True)
        header = research.runtime / "export" / MODEL_API_HEADER
        task = Path(__file__).parent / "prompts/task.md"
        staged = bool(research.configuration.experiment.stages)
        if staged:
            from .review import PROMPTS, PROMPT_FILES, prompt_inventory

            task = PROMPTS / "task.md"
            inventory = prompt_inventory()
            expected = research.configuration.experiment.prompt_sha256
            if expected and inventory != expected:
                raise ValueError("staged prompts changed after configuration was frozen")
            for name in PROMPT_FILES:
                publish_bytes(self.root / "references/prompts" / name, (PROMPTS / name).read_bytes())
            stage = research.active_stage
            if stage is None:
                raise ValueError("staged workspace needs an active stage")
            stage_prompt = "single_core.md" if stage.core_counts == (1,) else "multicore.md"
            publish_bytes(self.root / "references/stage.md", (PROMPTS / stage_prompt).read_bytes())
        publish_bytes(
            research.root / "workspace-bindings" / f"{iteration}-{role}.json",
            canonical_json(
                {
                    "candidate": candidate,
                    "history": history,
                    "api_sha256": file_sha256(header),
                    "task_sha256": file_sha256(task),
                    **({"prompt_sha256": inventory, "stage": stage.model_dump(mode="json"),
                        "review_context": review_context} if staged else {}),
                }
            ).encode(),
        )
        if not (self.root / "draft").exists():
            materialize(
                research.root / "candidates",
                candidate["candidate_id"],
                self.root / "draft",
                MODEL_FILES,
                maximum_bytes=research.resources.source_bytes,
            )
            # materialize intentionally writes ordinary model files. The
            # process policy grants writes only to a proposer's draft directory.
        for relative, data in (
            ("references/api.h", header.read_bytes()),
            ("references/task.md", task.read_bytes()),
        ):
            publish_bytes(self.root / relative, data)
        if review_context is not None:
            if role != "reviewer" or not staged:
                raise ValueError("comparison evidence belongs only to the staged reviewer")
            target = self.root / "references/incumbent"
            if not target.exists():
                materialize(research.root / "candidates", review_context["incumbent"]["candidate_id"],
                            target, MODEL_FILES, maximum_bytes=research.resources.source_bytes)
            publish_bytes(self.root / "references/comparison.json",
                          canonical_json(review_context["comparison"]).encode())
        for outcome in history:
            number = outcome["iteration"]
            summary = outcome["summary"]
            data = read_file(
                research.root, summary["path"], maximum_bytes=research.resources.source_bytes
            )
            if file_sha256(research.root / summary["path"]) != summary["sha256"]:
                raise ValueError("prior iteration summary changed")
            publish_bytes(self.root / f"references/history/{number}/summary.md", data)
            publish_bytes(
                self.root / f"references/history/{number}/result.json",
                canonical_json(outcome).encode(),
            )
            target = self.root / f"references/history/{number}/model"
            if not target.exists():
                materialize(
                    research.root / "candidates",
                    outcome["candidate"]["candidate_id"],
                    target,
                    MODEL_FILES,
                    maximum_bytes=research.resources.source_bytes,
                )
        (self.root / "notes").mkdir(exist_ok=True)

    def record(self, event):
        data = (canonical_json(to_jsonable_python(event)) + "\n").encode()
        # Concatenated gzip members preserve complete individual events. A file
        # lock keeps concurrent MCP status/file responses from interleaving.
        with self.events.open("ab") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(
                gzip.compress(data, compresslevel=self.research.resources.gzip_level, mtime=0)
            )
            stream.flush()
            os.fsync(stream.fileno())

    def set_phase(self, phase):
        staged = bool(self.research.configuration.experiment.stages)
        allowed = (("review", "review_format") if staged else ("review",)) if self.role == "reviewer" else (
            ("explore", "reflect") if staged else ("explore", "revise", "reflect"))
        if phase not in allowed:
            raise PermissionError("phase does not belong to this role")
        self.phase = phase

    def _working(self):
        if self.role != "proposer" or self.phase not in ("explore", "revise"):
            raise PermissionError("model editing and diagnostics are closed in this phase")

    def _candidate(self, candidate_id=None):
        if candidate_id is None:
            return self.snapshot()
        return verify_snapshot(
            self.research.root / "candidates",
            candidate_id,
            MODEL_FILES,
            maximum_bytes=self.research.resources.source_bytes,
        )

    def snapshot(self):
        return snapshot(
            self.root / "draft",
            self.research.root / "candidates",
            MODEL_FILES,
            maximum_bytes=self.research.resources.source_bytes,
        )

    def summary(self):
        data = read_file(
            self.root, "notes/summary.md", maximum_bytes=self.research.resources.source_bytes
        )
        if not data.decode().strip():
            raise ValueError("the model must write a non-empty summary.md")
        from .records import CampaignState

        recovery = CampaignState(self.research.root / "campaign.sqlite").get(
            f"operational-recovery:{self.iteration}"
        )
        suffix = "-recovered" if recovery is not None else ""
        target = self.research.root / "summaries" / f"iteration-{self.iteration}{suffix}.md"
        publish_bytes(target, data)
        return {"path": str(target.relative_to(self.research.root)), "sha256": file_sha256(target)}

    def grants(self):
        """Positive native-process roots, not an instruction to trust file modes."""
        reads = [self.root / name for name in ("references", "draft", "notes")]
        writes = []
        if self.role == "proposer":
            writes = [self.root / "notes"]
            if self.phase in ("explore", "revise"):
                writes.append(self.root / "draft")
        return {"read": reads, "write": writes}

    @recorded
    def files(self) -> dict:
        """List the source, references and notes provided to this session."""
        return {
            "files": sorted(
                str(p.relative_to(self.root))
                for name in ("draft", "references", "notes")
                for p in (self.root / name).rglob("*")
                if p.is_file() and not p.is_symlink()
            )
        }

    @recorded
    def read(self, path: str, offset: int = 0, characters: int = 65536) -> dict:
        """Read a Unicode character page of a session file; offsets are not bytes."""
        safe_name(path)
        if path.split("/")[0] not in ("draft", "references", "notes"):
            raise PermissionError("file is not in the session's visible roots")
        if offset < 0 or not 1 <= characters <= 262144:
            raise ValueError("invalid page; use 1 to 262144 characters and a nonnegative offset")
        text = read_file(
            self.root,
            path,
            maximum_bytes=self.research.resources.source_bytes,
            require_single_link=True,
        ).decode()
        part = text[offset : offset + characters]
        return {
            "text": part,
            "next_offset": offset + len(part),
            "eof": offset + len(part) >= len(text),
        }

    @recorded
    def write(self, path: str, text: str) -> dict:
        """Replace model source/parameters, or write notes and the final summary."""
        safe_name(path)
        allowed = (
            self.role == "proposer"
            and path.startswith("notes/")
            and self.phase in ("explore", "revise", "reflect")
        )
        if path in {"draft/" + name for name in MODEL_FILES.paths}:
            self._working()
            allowed = True
        if not allowed:
            raise PermissionError("file is not editable in this phase")
        data = text.encode()
        if len(data) > self.research.resources.source_bytes:
            raise ValueError("file exceeds the configured source/notes size guard")
        replace_file(self.root, path, data)
        return {"path": path, "bytes": len(data)}

    @recorded
    def build(self) -> dict:
        """Compile/check the current draft with the fixed model API and -O3."""
        self._working()
        return self.research.check(self.snapshot())

    @recorded
    def evaluate_training(self) -> dict:
        """Evaluate the current draft on the entire training cohort; never test data."""
        self._working()
        return self.research.train(self.snapshot())

    @recorded
    def synthetic(self, request: SyntheticRequest) -> dict:
        """Generate a controlled pattern; diagnostic only, with a simulation deadline."""
        self._working()
        return self.research.synthetic(
            self.snapshot(), request, self.research.configuration.experiment.synthetic_limits
        )

    @recorded
    def open_loop(self, workload: str) -> dict:
        """Replay training-oracle observations with a deadline; outstanding reads may be absent."""
        self._working()
        return self.research.replay(self.snapshot(), workload)

    def diagnostic_failures(self) -> list[dict]:
        """Also retain failures that finish after the proposer stops polling."""
        if not self.events.exists():
            return []
        with gzip.open(self.events, "rt") as stream:
            return [event["diagnostic_failure"] for line in stream
                    if (event := json.loads(line)).get("diagnostic_failure")]

    @recorded
    def inspect_training(
        self,
        workload: str,
        model: str = "candidate",
        candidate_id: str | None = None,
        trace: str | None = None,
        offset: int = 0,
        lines: int = 100,
    ) -> dict:
        """Read training statistics or a CSV line page; no arbitrary filesystem paths."""
        self._working()
        if model not in ("oracle", "candidate", *COMPARISONS):
            raise ValueError("unknown comparison model")
        if offset < 0 or not 1 <= lines <= 1000:
            raise ValueError("invalid line page")
        case = self.research.training_case(workload)
        candidate = self._candidate(candidate_id) if model == "candidate" else None
        observed = self.research.inspect_training(workload, model, candidate)
        if trace is None:
            return {"statistics": observed["observation"]}
        if trace not in case.observation_names:
            raise PermissionError("trace is not a training observation")
        # The evaluator stores protected, verified gzip files. A page need not
        # decode the suffix of an already checked immutable trace.
        with artifacts.open_text(self.research.trace_path(observed, trace)) as stream:
            page = list(islice(stream, offset, offset + lines + 1))
        return {
            "text": "".join(page[:lines]),
            "next_offset": offset + min(lines, len(page)),
            "eof": len(page) <= lines,
        }

    @recorded
    def inspect_input(self, workload: str, offset: int = 0, instructions: int = 100, core: int = 0) -> dict:
        """Decode any page of a training ChampSim input; offset counts instructions.

        Addresses here are virtual input addresses, not controller physical
        addresses. Pagination limits response size, not which instructions may
        be inspected. Validation/test inputs are never accepted by this tool.
        """
        self._working()
        case = self.research.training_case(workload)
        members = (case, *getattr(case, "companions", ()))
        if type(core) is not int or not 0 <= core < len(members):
            raise ValueError("core is outside this training case")
        case = members[core]
        if case.identity()["frontend"] != "champsim":
            raise ValueError("instruction decoding requires a ChampSim training case")
        if offset < 0 or not 1 <= instructions <= 1000:
            raise ValueError("use a nonnegative instruction offset and a page of 1 to 1000")
        record = struct.Struct("<QBB2B4B2Q4Q")
        if case.instruction_inventory["record_bytes"] != record.size:
            raise ValueError("input is not the standard 64-byte ChampSim instruction format")
        opener = gzip.open if case.instruction_inventory.get("compression") == "gzip" else lzma.open
        total = case.instruction_inventory["instructions"]
        if offset >= total:
            return {"instructions": [], "next_offset": offset, "eof": True}
        with opener(case.payload.source, "rb") as stream:
            stream.seek(offset * record.size)
            data = stream.read(min(instructions, total - offset) * record.size)
        rows = []
        for index, fields in enumerate(record.iter_unpack(data), offset):
            rows.append({
                "instruction": index, "ip": fields[0],
                "is_branch": fields[1], "branch_taken": fields[2],
                "destination_registers": fields[3:5], "source_registers": fields[5:9],
                "destination_memory": fields[9:11], "source_memory": fields[11:15],
            })
        return {"instructions": rows, "next_offset": offset + len(rows),
                "eof": offset + len(rows) >= total}

    def methods(self):
        """The sole tool allowlist, shared by native and scripted transports."""
        methods = [self.files, self.read]
        if self.role == "proposer":
            methods.append(self.write)
            if self.phase in ("explore", "revise"):
                methods += [self.build, self.evaluate_training, self.inspect_training]
                if self.research.configuration.experiment.evaluation.champsim:
                    methods.append(self.inspect_input)
                features = self.research.configuration.experiment.features
                if features.synthetic_diagnostics:
                    methods.append(self.synthetic)
                if features.open_loop_diagnostics:
                    methods.append(self.open_loop)
        return {method.__name__: method for method in methods}
