"""DRAM measurements and diagnostics on CHIA tasks, independent of model backend.

The campaign chooses when to train or test. This module owns the scientific
cases and source-bound native jobs; it never calls an LLM or promotes a model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import ray
from chia.base.ChiaFunction import ChiaFunction, get
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from tools.chia_loop.recovery import exclusive_lock
from tools.eval import controller_replay, synthetic

from .archive import Payload, describe_payload
from .candidate import BuildFailed, BuildLimits, compile_snapshot, runtime_inputs
from .config import CampaignConfig, SyntheticLimits
from .diagnostic_cases import ReplayCase, SyntheticCase, SyntheticRequest
from .evaluation import (
    CandidateBuild,
    IsolatedRamulator,
    MeasurementFailed,
    SimulationLimits,
    measure_native,
)
from .export import capture_sources
from .identity import canonical_json, digest_json, file_sha256
from .measurement_reports import compare_simpleo3, compare_transfer, verify_native_measurement
from .scoring import aggregate
from .snapshots import ModelFiles, publish_bytes, snapshot

COMPARISONS = ("fixedlat", "md1", "wmg1", "mess")
MODEL_FILES = ModelFiles(("model.cpp",), "parameters.json")


class CandidateEvaluationFailed(RuntimeError):
    def __init__(self, failures):
        self.failures = failures
        super().__init__("candidate evaluation failed; measurements are unavailable")


def _read(path):
    return json.loads(path.read_text())


@ChiaFunction(num_cpus=1, max_retries=0)
def build_model(runtime: Path, root: Path, candidate: dict, limits: BuildLimits) -> dict:
    directory = root / "builds" / candidate["candidate_id"]
    with exclusive_lock(directory / "job.lock"):
        result = directory / "result.json"
        if result.exists():
            receipt = _read(result)
        else:
            attempt = directory / uuid4().hex
            try:
                built = compile_snapshot(
                    runtime,
                    root / "candidates",
                    candidate["candidate_id"],
                    MODEL_FILES,
                    attempt,
                    limits=limits,
                )
            except BuildFailed as exc:
                built = exc.receipt
            receipt = {
                "candidate": candidate,
                "passed": built["passed"],
                "directory": str(attempt.relative_to(root)),
                "receipt_sha256": file_sha256(attempt / "build.json"),
                "error": built.get("error"),
            }
            publish_bytes(result, canonical_json(receipt).encode())
        if receipt["passed"]:
            candidate_build(root, receipt).verify(runtime, limits.source_bytes)
        elif file_sha256(root / receipt["directory"] / "build.json") != receipt["receipt_sha256"]:
            raise ValueError("failed build evidence changed")
        if receipt["candidate"] != candidate:
            raise ValueError("build receipt belongs to another submitted source")
        return receipt


def candidate_build(root: Path, receipt: dict) -> CandidateBuild:
    return CandidateBuild(
        root / "candidates",
        receipt["candidate"]["candidate_id"],
        MODEL_FILES,
        root / receipt["directory"],
        receipt["receipt_sha256"],
    )


@ChiaFunction(num_cpus=1, max_retries=0)
def measurement(
    runtime: Path,
    root: Path,
    case,
    model: str,
    limits: SimulationLimits,
    build: dict | None,
    curve: Payload | None,
    host,
) -> dict:
    identity = {
        "runtime": file_sha256(runtime / "runtime_manifest.json"),
        "case": case.identity(),
        "model": model,
        "limits": asdict(limits),
        "candidate": build["candidate"] if build else None,
        "curve": asdict(curve.member) if curve else None,
        "host": host.identity(),
    }
    directory = root / "measurements" / digest_json(identity)
    with exclusive_lock(directory / "job.lock"):
        result = directory / "result.json"
        if result.exists():
            receipt = _read(result)
            verify_native_measurement(
                root / receipt["directory"],
                receipt["receipt_sha256"],
                frontend=case.identity()["frontend"],
                observations=case.observation_names,
            )
            return receipt
        attempt = directory / uuid4().hex
        # Failure evidence is written by the shared executor. It is never
        # published as a successful job; the caller can inspect the exception.
        observed = measure_native(
            runtime,
            case,
            model,
            attempt,
            limits=limits,
            candidate=candidate_build(root, build) if build else None,
            mess_curve=curve,
            host=host,
        )
        receipt = {
            "identity": identity,
            "directory": str(attempt.relative_to(root)),
            "receipt_sha256": file_sha256(attempt / "measurement.json"),
            "observation": observed,
        }
        publish_bytes(result, canonical_json(receipt).encode())
        return receipt


@dataclass
class DramResearch:
    configuration: CampaignConfig
    root: Path
    runtime: Path
    cases: dict[str, tuple]
    curve: Payload
    node_id: str
    session_factory: object = None
    hosts: dict | None = None
    _inspection: dict = field(default_factory=dict, init=False, repr=False)

    def __getstate__(self):
        # CHIA sends evaluation tools to a worker. The model launcher belongs
        # only to the campaign driver; never pickle provider transports/login
        # handles into a tool actor just because it needs evaluation methods.
        return {**self.__dict__, "session_factory": None, "_inspection": {}}

    def __post_init__(self):
        if self.configuration.execution != "native":
            raise ValueError("native DRAM evaluation cannot be labeled a numerical fixture")
        self.root, self.runtime = self.root.absolute(), self.runtime.absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        if not ray.is_initialized() or ray.cluster_resources().get("CPU", 0) > self.resources.cpus:
            raise ValueError("use a CHIA/Ray cluster within the configured CPU envelope")
        self.hosts = self.hosts or {}
        evaluation = self.configuration.experiment.evaluation
        expected = {
            "training": evaluation.simpleo3.training,
            "test": evaluation.simpleo3.test,
            **{
                name: group.workloads
                for name, group in (
                    ("champsim", evaluation.transfer.champsim),
                    ("gem5", evaluation.transfer.gem5),
                )
                if group is not None
            },
        }
        if set(self.cases) != set(expected):
            raise ValueError("native cases differ from the configured evaluation groups")
        for group, names in expected.items():
            if tuple(case.workload for case in self.cases[group]) != names:
                raise ValueError("native cases differ from the configured workload order")
            if group in ("training", "test") and any(
                case.stage != group
                or case.instructions_per_core != evaluation.simpleo3.instructions_per_core
                for case in self.cases[group]
            ):
                raise ValueError("native cases changed the stage or instruction window")
            if group not in ("training", "test"):
                if group not in self.hosts or any(c.frontend != group for c in self.cases[group]):
                    raise ValueError("transfer cases require their declared frontend host")
                if group == "champsim" and any(
                    c.warmup_instructions != evaluation.transfer.champsim.warmup_instructions
                    or c.roi_instructions != evaluation.transfer.champsim.roi_instructions
                    for c in self.cases[group]
                ):
                    raise ValueError("ChampSim cases changed the configured instruction windows")
        train_hashes = {
            p.member.logical_sha256
            for c in self.cases["training"]
            for p in c.input_files().values()
        }
        test_hashes = {
            p.member.logical_sha256 for c in self.cases["test"] for p in c.input_files().values()
        }
        if train_hashes & test_hashes:
            raise ValueError("training and testing share an input trace")
        runtime_inputs(self.runtime)
        source = capture_sources(
            Path(__file__).resolve().parents[3], self.runtime, self.root / "source"
        )
        # This cheap descriptor binding is checked on every construction/resume.
        # Payload decoders verify the actual bytes when ingesting/staging inputs.
        publish_bytes(
            self.root / "native-inputs.json",
            canonical_json(
                {
                    "configuration": self.configuration.model_dump(mode="json"),
                    "runtime": file_sha256(self.runtime / "runtime_manifest.json"),
                    "source_sha256": source["sha256"],
                    "cases": {k: [c.identity() for c in v] for k, v in self.cases.items()},
                    "curve": asdict(self.curve.member),
                    "resources": self.resources.model_dump(),
                    "hosts": {k: h.identity() for k, h in self.hosts.items()},
                }
            ).encode(),
        )

    @property
    def resources(self):
        return self.configuration.run.resources

    @property
    def limits(self):
        r = self.resources
        return SimulationLimits(
            r.simulation_timeout_seconds, r.memory_bytes, r.file_bytes, r.source_bytes, r.gzip_level
        )

    def _dispatch(self, function, *args):
        return function.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(self.node_id, soft=False)
        ).remote(*args)

    def prepare(self):
        workspace = self.root / "seed"
        publish_bytes(
            workspace / "model.cpp",
            (self.runtime / "runtime-source/tools/chia_loop/model/seed.cpp").read_bytes(),
        )
        publish_bytes(workspace / "parameters.json", b"{}")
        seed = snapshot(
            workspace,
            self.root / "candidates",
            MODEL_FILES,
            maximum_bytes=self.resources.source_bytes,
        )
        comparisons = self.cohort(None, "training", COMPARISONS)
        return {"seed": seed, "comparisons": comparisons}

    def check(self, candidate):
        r = self.resources
        return get(
            self._dispatch(
                build_model,
                self.runtime,
                self.root,
                candidate,
                BuildLimits(
                    1, r.build_timeout_seconds, r.source_bytes, r.memory_bytes, r.file_bytes
                ),
            )
        )

    def measure(self, case, model, build=None):
        frontend = case.identity()["frontend"]
        host = self.hosts.get(frontend, IsolatedRamulator())
        return self._dispatch(
            measurement,
            self.runtime,
            self.root,
            case,
            model,
            self.limits,
            build if model == "candidate" else None,
            self.curve if model == "mess" else None,
            host,
        )

    def cohort(self, candidate, group, models=("candidate",)):
        build = self.check(candidate) if "candidate" in models else None
        if build and not build["passed"]:
            raise ValueError("candidate failed its source build; inspect check() feedback")
        jobs = {
            (case.workload, model): self.measure(case, model, build)
            for case in self.cases[group]
            for model in ("oracle", *models)
        }
        measured, errors = {}, []
        for key, ref in jobs.items():
            try:
                measured[key] = get(ref)
            except Exception as exc:
                errors.append((key, exc))
        if errors:
            # Wait for every submitted job before returning. Oracle/baseline,
            # input-integrity and worker failures are infrastructure problems,
            # not a bad model score or permission to drop a workload.
            for (_, model), exc in errors:
                if (
                    model != "candidate"
                    or not isinstance(exc, MeasurementFailed)
                    or exc.receipt.get("archive_error")
                ):
                    raise exc
            raise CandidateEvaluationFailed(
                {workload: exc.receipt for (workload, _), exc in errors}
            )
        stage = group if group in ("training", "test") else "transfer"
        reports = {}
        for model in models:
            rows = {}
            for case in self.cases[group]:
                oracle, value = (measured[case.workload, name] for name in ("oracle", model))
                compare = compare_simpleo3 if stage != "transfer" else compare_transfer
                rows[case.workload] = compare(
                    self.root / oracle["directory"],
                    self.root / value["directory"],
                    oracle_receipt_sha256=oracle["receipt_sha256"],
                    model_receipt_sha256=value["receipt_sha256"],
                    minimum_oracle_owner_reads=(
                        self.configuration.experiment.evaluation.simpleo3.minimum_oracle_owner_reads
                    ),
                    **({"frontend": group} if stage == "transfer" else {}),
                )
            reports[model] = aggregate(
                rows, expected_workloads=[c.workload for c in self.cases[group]], stage=stage
            )
        return reports

    def train(self, candidate):
        receipt = {
            "candidate": candidate,
            "execution": "native",
            "evaluation_sha256": digest_json(
                self.configuration.experiment.evaluation.model_dump(mode="json")
            ),
        }
        try:
            receipt["measurement"] = self.cohort(candidate, "training")["candidate"]
        except CandidateEvaluationFailed as exc:
            receipt.update(measurement=None, failure=exc.failures)
        return receipt

    def postrun(self, candidate):
        return {
            "candidate": candidate,
            "execution": "native",
            "reports": {
                group: self.cohort(candidate, group, ("candidate", *COMPARISONS))
                for group in self.cases
                if group != "training"
            },
        }

    def session(self, iteration, role, candidate, history):
        if self.session_factory is None:
            raise RuntimeError("no native/scripted session binding was supplied")
        return self.session_factory(self, iteration, role, candidate, history)

    def training_case(self, workload):
        for case in self.cases["training"]:
            if case.workload == workload:
                return case
        raise PermissionError("workload is not available for training diagnostics")

    def inspect_training(self, workload, model, candidate=None):
        case = self.training_case(workload)
        key = (workload, model, candidate["candidate_id"] if candidate else None)
        if key not in self._inspection:
            build = self.check(candidate) if model == "candidate" else None
            self._inspection[key] = get(self.measure(case, model, build))
        # Native jobs verify the complete compressed evidence before returning.
        # This session-local view points to protected, immutable files; later
        # pages do not launch a job just to repeat that full-file validation.
        return self._inspection[key]

    def synthetic(self, candidate, request: SyntheticRequest, limits: SyntheticLimits):
        if not self.configuration.experiment.features.synthetic_diagnostics:
            raise PermissionError("synthetic diagnostics are disabled")
        case = SyntheticCase.create(request, limits)
        build = self.check(candidate)
        measured = {name: self.measure(case, name, build) for name in ("oracle", "candidate")}
        measured = {name: get(ref) for name, ref in measured.items()}
        records = {name: result["observation"] for name, result in measured.items()}
        populations = {
            name: synthetic.read_population(
                self.trace_path(result, "controller.csv.ch0"),
                case.parameters,
                result["observation"]["frontend_stats"],
            )
            for name, result in measured.items()
        }
        return {
            "diagnostic_only": True,
            "eligible_for_promotion": False,
            **synthetic.summarize(
                populations["oracle"], populations["candidate"], records, limits.maximum_trace_rows
            ),
        }

    def replay(self, candidate, workload):
        if not self.configuration.experiment.features.open_loop_diagnostics:
            raise PermissionError("open-loop diagnostics are disabled")
        training = self.training_case(workload)
        oracle = get(self.measure(training, "oracle"))
        population = controller_replay.oracle_population(
            self.trace_path(oracle, "controller.csv.ch0"),
            oracle["observation"]["controller_stats"],
        )
        directory = self.root / "replay-inputs" / oracle["receipt_sha256"]
        with exclusive_lock(directory / "job.lock"):
            path = directory / "arrivals.csv.gz"
            if not path.exists():
                # Dataframe compression uses gzip directly; no expanded trace
                # survives between diagnostic calls.
                temporary = directory / (uuid4().hex + ".gz")
                population.to_csv(
                    temporary,
                    columns=list(controller_replay.ARRIVAL_COLUMNS),
                    header=True,
                    index=False,
                    compression={"method": "gzip", "mtime": 0},
                )
                describe_payload(temporary, "arrivals.csv.gz", codec="gzip")
                temporary.rename(path)
            arrivals = describe_payload(path, "arrivals.csv.gz", codec="gzip")
        case = ReplayCase(
            workload,
            arrivals,
            {"sha256": oracle["receipt_sha256"]},
            len(training.traces),
            int((population["type"] == 0).sum()),
            int((population["type"] == 1).sum()),
        )
        replay = get(self.measure(case, "candidate", self.check(candidate)))
        return controller_replay.summarize(
            population,
            self.trace_path(replay, "replay.csv"),
            replay["observation"]["batch_stats"],
        )

    def trace_path(self, result, name):
        return self.root / result["directory"] / result["observation"]["traces"][name]["name"]
