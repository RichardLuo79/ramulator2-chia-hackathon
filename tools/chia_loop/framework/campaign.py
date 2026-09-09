"""The common research workflow. Backends supply sessions, never a second loop."""

from __future__ import annotations

from pathlib import Path
from typing import ContextManager, Protocol

from tools.chia_loop.recovery import exclusive_lock

from .config import CampaignConfig
from .identity import digest_json
from .records import CampaignState
from .scoring import aggregate, should_promote, valid_objectives


class Session(Protocol):
    """One native conversation with role/phase-scoped tools and retained evidence."""

    def turn(self, phase: str, inputs: dict) -> dict: ...
    def snapshot(self) -> dict: ...
    def summary(self) -> dict: ...


class Research(Protocol):
    """The DRAM application, implemented once and shared by all native adapters.

    Methods return compact receipts referencing source/evidence on disk. Agents
    do not receive this object: they see only the session's approved tool view.
    Implementations dispatch expensive work through CHIA and preserve native
    continuation before returning from a model turn.
    """

    configuration: CampaignConfig

    def prepare(self) -> dict: ...
    def check(self, candidate: dict) -> dict: ...
    def train(self, candidate: dict) -> dict: ...
    def postrun(self, candidate: dict) -> dict: ...
    def session(
        self, iteration: int, role: str, candidate: dict, history: list[dict]
    ) -> ContextManager[Session]: ...


class Campaign:
    def __init__(self, root: Path, configuration: CampaignConfig, research: Research):
        if research.configuration != configuration:
            raise ValueError("research services use different campaign settings")
        self.root = root.absolute()
        self.config = configuration
        self.research = research
        self.state = CampaignState(self.root / "campaign.sqlite")

    def _step(self, key, inputs, action, *, repeatable=False):
        run = self.config.run
        return self.state.step(
            key,
            inputs,
            action,
            maximum_attempts=run.maximum_attempts,
            retry_delay_seconds=run.retry_delay_seconds,
            repeat_after_interruption=repeatable,
        )

    def _training(self, candidate: dict) -> dict:
        result = self._step(
            "training:" + candidate["candidate_id"],
            candidate,
            lambda: self.research.train(candidate),
            repeatable=True,
        )
        simple = self.config.experiment.evaluation.simpleo3
        for key, expected in {
            "candidate": candidate,
            "execution": self.config.execution,
            "evaluation_sha256": digest_json(
                self.config.experiment.evaluation.model_dump(mode="json")
            ),
        }.items():
            if result.get(key) != expected:
                raise ValueError(f"training receipt has wrong {key}")
        measurement = result["measurement"]
        if measurement is None:
            # A model that compiles can still violate the runtime contract.
            # Keep its diagnostics and let the next iteration learn from them;
            # do not turn a bad candidate into an infrastructure deadlock.
            if not result.get("failure"):
                raise ValueError("missing training measurements need a recorded failure")
            return result
        for row in measurement["workloads"].values():
            if row.get("minimum_oracle_owner_reads") != simple.minimum_oracle_owner_reads:
                raise ValueError("training measurement changed the traffic guard")
            if row.get("observation") != {
                "scope": "simpleo3_logical_llc",
                "timebase": "simpleo3_frontend_cycles",
            }:
                raise ValueError("training measurement changed the request population")
        recomputed = aggregate(
            measurement["workloads"], expected_workloads=list(simple.training), stage="training"
        )
        if measurement != recomputed:
            raise ValueError("training summary differs from its workload measurements")
        return result

    def _check(self, candidate):
        result = self._step(
            "check:" + candidate["candidate_id"],
            candidate,
            lambda: self.research.check(candidate),
            repeatable=True,
        )
        if result.get("candidate") != candidate or type(result.get("passed")) is not bool:
            raise ValueError("mechanical check is not bound to the submitted source")
        return result

    def _iteration(self, number: int, incumbent: dict, history: list[dict]) -> dict:
        key = f"iteration:{number}"
        existing = self.state.get(key)
        if existing is not None:
            if existing["parent"] != incumbent["candidate"]:
                raise ValueError("committed iteration has a different parent")
            return existing
        task = self.config.experiment.agent_view()
        with self.research.session(number, "proposer", incumbent["candidate"], history) as session:
            self._step(
                key + ":explore",
                {"task": task, "incumbent": incumbent, "history": history},
                lambda: session.turn("explore", {"task": task, "training": incumbent}),
            )
            draft = self._step(key + ":draft", {}, session.snapshot, repeatable=True)
            critique = None
            if self.config.experiment.semantic_llm_check:

                def review():
                    with self.research.session(number, "reviewer", draft, []) as reviewer:
                        return reviewer.turn("review", {"candidate": draft})

                critique = self._step(key + ":review", {"candidate": draft}, review)
                self._step(
                    key + ":revise",
                    {"critique": critique},
                    lambda: session.turn("revise", {"critique": critique["text"]}),
                )
            candidate = self._step(key + ":final", {}, session.snapshot, repeatable=True)
            check = self._check(candidate)
            training = self._training(candidate) if check["passed"] else None
            promoted = bool(training and training["measurement"]) and should_promote(
                training["measurement"], incumbent["measurement"], mechanically_valid=True
            )
            outcome = {
                "iteration": number,
                "parent": incumbent["candidate"],
                "candidate": candidate,
                "check": check,
                "training": training,
                "promoted": promoted,
                "selected": training if promoted else incumbent,
                "critique": critique,
            }
            # The session changes its tool grants for reflection: it may write
            # summary.md, but cannot edit the now-evaluated model or run more jobs.
            self._step(key + ":reflect", outcome, lambda: session.turn("reflect", outcome))
            summary = self._step(key + ":summary", {}, session.summary, repeatable=True)
            return self.state.save(key, {**outcome, "summary": summary})

    def run_search(self) -> dict:
        """Resume this campaign and freeze one selection at the iteration guard."""
        with exclusive_lock(self.root / "campaign.lock"):
            self.state.save("configuration", self.config.model_dump(mode="json"))
            frozen = self.state.get("selection")
            if frozen is not None:
                return frozen
            prepared = self._step("prepare", {}, self.research.prepare, repeatable=True)
            seed = prepared["seed"]
            if not self._check(seed)["passed"]:
                raise ValueError("initial seed fails mechanical checks")
            incumbent = self._training(seed)
            valid_objectives(incumbent["measurement"])
            history = []
            for number in range(1, self.config.run.maximum_iterations + 1):
                outcome = self._iteration(number, incumbent, history)
                incumbent = outcome["selected"]
                history.append(outcome)
            return self.state.save(
                "selection",
                {
                    "candidate": incumbent["candidate"],
                    "training": incumbent,
                    "iterations": len(history),
                    "execution": self.config.execution,
                },
            )

    def evaluate(self) -> dict:
        """Frozen held-out/transfer evaluation; no archive or model call required."""
        with exclusive_lock(self.root / "campaign.lock"):
            self.state.save("configuration", self.config.model_dump(mode="json"))
            frozen = self.state.get("selection")
            if frozen is None:
                raise ValueError("freeze the training selection before held-out evaluation")
            return self._step(
                "postrun",
                frozen,
                lambda: self.research.postrun(frozen["candidate"]),
                repeatable=True,
            )
