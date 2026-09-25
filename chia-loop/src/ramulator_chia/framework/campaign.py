"""The common research workflow. Backends supply sessions, never a second loop."""

from __future__ import annotations

from pathlib import Path
from typing import ContextManager, Protocol

from ramulator_chia.recovery import exclusive_lock

from .config import CampaignConfig, diagnostic_policy, preserved_configuration
from .identity import digest_json
from .records import CampaignState, RecordConflict
from .scoring import aggregate, should_promote, valid_objectives, validation_view
from . import review


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
    def validate(self, candidate: dict) -> dict: ...
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
        self.active_stage = None

    def _save_configuration(self):
        try:
            original = preserved_configuration(self.config, self.state.get("configuration"))
        except ValueError as exc:
            raise RecordConflict(str(exc)) from exc
        self.state.save("configuration", original)
        self.state.save("diagnostic-policy", diagnostic_policy(self.config, original))

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

    def _measurement(self, candidate: dict, stage: str) -> dict:
        prefix = f"stage:{self.active_stage.name}:" if self.active_stage else ""
        result = self._step(
            prefix + stage + ":" + candidate["candidate_id"],
            candidate,
            lambda: (
                self.research.train(candidate) if stage == "training"
                else self.research.validate(candidate)
            ),
            repeatable=True,
        )
        evaluation = self.config.experiment.evaluation
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
        if self.active_stage:
            if result.get("campaign_stage") != self.active_stage.model_dump(mode="json"):
                raise ValueError("measurement belongs to another campaign stage")
            if measurement is None:
                if not result.get("failure"):
                    raise ValueError("missing measurements need a recorded failure")
                return result
            rows = {name: row for group in measurement["groups"].values()
                    for name, row in group["workloads"].items()}
            if any(row.get("minimum_oracle_owner_reads") != evaluation.primary.minimum_oracle_owner_reads
                   for row in rows.values()):
                raise ValueError("staged measurement changed the traffic guard")
            recomputed = review.grouped_measurement(
                rows, names=self.config.experiment.case_names(stage, self.active_stage),
                core_counts=self.active_stage.core_counts, cases=evaluation.champsim.cases, stage=stage,
            )
            if measurement != recomputed:
                raise ValueError("staged summary differs from its case measurements")
            return result
        if measurement is None:
            # A model that compiles can still violate the runtime contract.
            # Keep its diagnostics and let the next iteration learn from them;
            # do not turn a bad candidate into an infrastructure deadlock.
            if not result.get("failure"):
                raise ValueError("missing training measurements need a recorded failure")
            return result
        for row in measurement["workloads"].values():
            if row.get("minimum_oracle_owner_reads") != evaluation.primary.minimum_oracle_owner_reads:
                raise ValueError("training measurement changed the traffic guard")
            if row.get("observation") != evaluation.observation:
                raise ValueError("training measurement changed the request population")
        recomputed = aggregate(
            measurement["workloads"], expected_workloads=list(getattr(evaluation.primary, stage)),
            stage=stage, request_objective=evaluation.request_objective,
        )
        if measurement != recomputed:
            raise ValueError("training summary differs from its workload measurements")
        return result

    def _training(self, candidate):
        return self._measurement(candidate, "training")

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
        if self.active_stage:
            return self._reviewed_iteration(number, incumbent, history)
        task = self.config.experiment.agent_view()
        validation_names = self.config.experiment.evaluation.validation
        incumbent_validation = (
            self._measurement(incumbent["candidate"], "validation") if validation_names else None
        )
        explore = {"task": task, "training": incumbent}
        if incumbent_validation is not None:
            explore["validation"] = validation_view(incumbent_validation, validation_names)
        with self.research.session(number, "proposer", incumbent["candidate"], history) as session:
            self._step(
                key + ":explore",
                {"task": task, "incumbent": incumbent, "history": history},
                lambda: session.turn("explore", explore),
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
            validation = (
                self._measurement(candidate, "validation")
                if validation_names and check["passed"] else None
            )
            validation_gate = {}
            if self.config.experiment.validation_non_worsening:
                validation_gate = {
                    "candidate_validation": validation["measurement"] if validation else None,
                    "incumbent_validation": incumbent_validation["measurement"],
                }
            promoted = bool(training and training["measurement"]) and should_promote(
                training["measurement"], incumbent["measurement"], mechanically_valid=True,
                **validation_gate,
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
            if validation_names:
                outcome["validation"] = validation_view(validation or {}, validation_names)
            # The session changes its tool grants for reflection: it may write
            # summary.md, but cannot edit the now-evaluated model or run more jobs.
            self._step(key + ":reflect", outcome, lambda: session.turn("reflect", outcome))
            summary = self._step(key + ":summary", {}, session.summary, repeatable=True)
            return self.state.save(key, {**outcome, "summary": summary})

    def _reviewed_iteration(self, number, incumbent, history):
        """One submission, one final review; all providers use this same path."""
        key = f"iteration:{number}"
        experiment = self.config.experiment
        task = experiment.agent_view(self.active_stage)
        train_names = experiment.case_names("training", self.active_stage)
        # Use the full validation ordering so aliases do not change at round 11.
        val_names = experiment.evaluation.validation
        incumbent_validation = self._measurement(incumbent["candidate"], "validation")
        anchor = self.state.get("stage-entry:" + self.active_stage.name)

        def scores(training, validation):
            return {
                "training": review.score_view(training or {}, train_names, anonymous=False),
                "validation": review.score_view(validation or {}, val_names, anonymous=True),
            }

        explore = {
            "task": task,
            "incumbent": scores(incumbent, incumbent_validation),
            "stage_entry": scores(anchor["training"], anchor["validation"]),
        }
        recovery = self.state.get(f"operational-recovery:{number}")
        if recovery is not None:
            # An operator must first preserve and supersede an infrastructure-
            # blocked outcome under the campaign lease. This is not permission
            # to reroll a reviewer decision or disclose private recovery evidence.
            original = self.state.get(recovery["superseded_iteration_key"])
            if (recovery.get("reason") != "compiler_identity_changed"
                    or original is None or original["iteration"] != number
                    or original["selection_reason"] != "unchanged_candidate"
                    or original["promotion_review"] is not None
                    or original["candidate"] != incumbent["candidate"]):
                raise RecordConflict("recovery requires an unchanged, unreviewed blocked round")
            explore["operational_recovery"] = (
                "The user authorized recovery of this same round after a compiler "
                "infrastructure failure prevented evaluation of a new draft. The "
                "original compiler has been restored and verified. Your earlier "
                "unchanged submission and reflection are preserved as superseded "
                "history, not an accuracy judgment on the untested draft. Continue "
                "this conversation using your saved notes and the normal training "
                "tools. Editing is reopened for this recovery only; this is not an "
                "additional round. Submit one final candidate for the usual "
                "independent review and reflection. Test evidence remains unavailable."
            )
        explore_identity = {"task": task, "incumbent": incumbent, "history": history}
        if recovery is not None:
            explore_identity["operational_recovery_sha256"] = digest_json(recovery)
        with self.research.session(number, "proposer", incumbent["candidate"], history) as session:
            self._step(
                key + ":explore",
                explore_identity,
                lambda: session.turn("explore", explore),
            )
            candidate = self._step(key + ":final", {}, session.snapshot, repeatable=True)
            check = self._check(candidate)
            training = self._training(candidate) if check["passed"] else None
            validation = self._measurement(candidate, "validation") if check["passed"] else None
            reason, verdict = None, None
            if (
                not check["passed"]
                or not review.eligible(training)
                or not review.eligible(validation)
            ):
                reason = "mechanical_or_measurement_failure"
            elif candidate == incumbent["candidate"]:
                reason = "unchanged_candidate"
            else:
                comparison = {
                    "task": task,
                    "candidate": scores(training, validation),
                    "incumbent": scores(incumbent, incumbent_validation),
                    "stage_entry": scores(anchor["training"], anchor["validation"]),
                    "prior_decisions": [
                        {
                            "iteration": row["iteration"],
                            "stage": row["stage"],
                            "decision": row["promotion_review"],
                        }
                        for row in history
                    ],
                }
                context = {"incumbent": incumbent["candidate"], "comparison": comparison}
                # Completed decisions are reused verbatim, even after a reflection failure.
                verdict = self.state.get(key + ":decision")
                if verdict is None:
                    with self.research.session(
                        number, "reviewer", candidate, [], review_context=context
                    ) as judge:
                        raw = self._step(
                            key + ":review", context, lambda: judge.turn("review", comparison)
                        )
                        try:
                            decision = review.parse_decision(raw["text"])
                        except (ValueError, TypeError):
                            repair = {
                                "instruction": "Format-only repair; preserve your original judgment.",
                                "previous_response": raw["text"],
                            }
                            raw = self._step(
                                key + ":review-format",
                                repair,
                                lambda: judge.turn("review_format", repair),
                            )
                            decision = review.parse_decision(
                                raw["text"], repair_of=repair["previous_response"]
                            )
                        verdict = self.state.save(
                            key + ":decision",
                            {
                                **decision,
                                "candidate": candidate,
                                "incumbent": incumbent["candidate"],
                                "comparison_sha256": digest_json(comparison),
                                "receipt": raw,
                            },
                        )
            promoted = bool(verdict and verdict["decision"] == "promote")
            outcome = {
                "iteration": number,
                "stage": self.active_stage.name,
                "parent": incumbent["candidate"],
                "candidate": candidate,
                "check": check,
                "training": training,
                "promoted": promoted,
                "selected": training if promoted else incumbent,
                "validation": review.score_view(validation or {}, val_names, anonymous=True),
                "promotion_review": verdict,
                "selection_reason": reason,
            }
            reflection = {
                "iteration": number,
                "stage": self.active_stage.name,
                "scores": scores(training, validation),
                "promoted": promoted,
                "promotion_review": verdict,
                "selection_reason": reason,
                "check": check,
            }
            self._step(key + ":reflect", outcome, lambda: session.turn("reflect", reflection))
            summary = self._step(key + ":summary", {}, session.summary, repeatable=True)
            return self.state.save(key, {**outcome, "summary": summary})

    def _activate(self, stage):
        self.active_stage = stage
        self.research.activate_stage(stage)

    def _enter_stage(self, stage, candidate):
        self._activate(stage)
        training = self._training(candidate)
        validation = self._measurement(candidate, "validation")
        if not review.eligible(training) or not review.eligible(validation):
            raise ValueError(
                "stage-entry model requires valid complete training and validation measurements"
            )
        return self.state.save(
            "stage-entry:" + stage.name,
            {
                "candidate": candidate,
                "training": training,
                "validation": validation,
            },
        )

    def run_search(self, *, before_round=None) -> dict:
        """Resume this campaign and freeze one selection at the iteration guard."""
        with exclusive_lock(self.root / "campaign.lock"):
            self._save_configuration()
            if self.config.experiment.stages:
                inventory = review.prompt_inventory()
                expected = self.config.experiment.prompt_sha256
                if expected and expected != inventory:
                    raise ValueError("reviewed prompt files differ from the campaign configuration")
                self.state.save("prompt_inventory", inventory)
                self._activate(self.config.experiment.stages[0])
            frozen = self.state.get("selection")
            if frozen is not None:
                return frozen
            prepared = self._step("prepare", {}, self.research.prepare, repeatable=True)
            seed = prepared["seed"]
            if not self._check(seed)["passed"]:
                raise ValueError("initial seed fails mechanical checks")
            incumbent = self._training(seed)
            if self.active_stage:
                self._enter_stage(self.active_stage, seed)
            else:
                valid_objectives(incumbent["measurement"])
            if self.config.experiment.evaluation.validation and not self.active_stage:
                initial_validation = self._measurement(seed, "validation")
                valid_objectives(initial_validation["measurement"], stage="validation")
            history = []
            endpoints = {}
            for number in range(1, self.config.run.maximum_iterations + 1):
                stage = self.config.experiment.stage_for(number)
                if before_round is not None and self.state.get(f"iteration:{number}") is None:
                    before_round(number, stage)
                if stage != self.active_stage:
                    incumbent = self._enter_stage(stage, incumbent["candidate"])["training"]
                outcome = self._iteration(number, incumbent, history)
                incumbent = outcome["selected"]
                history.append(outcome)
                if stage and (
                    number == self.config.run.maximum_iterations
                    or self.config.experiment.stage_for(number + 1) != stage
                ):
                    endpoints[stage.name] = self.state.save(
                        "stage-selection:" + stage.name,
                        {
                            "candidate": incumbent["candidate"],
                            "training": incumbent,
                            "iteration": number,
                        },
                    )
            return self.state.save(
                "selection",
                {
                    "candidate": incumbent["candidate"],
                    "training": incumbent,
                    "iterations": len(history),
                    "execution": self.config.execution,
                    **({"stage_selections": endpoints} if endpoints else {}),
                },
            )

    def evaluate(self) -> dict:
        """Frozen held-out/transfer evaluation; no archive or model call required."""
        with exclusive_lock(self.root / "campaign.lock"):
            self._save_configuration()
            frozen = self.state.get("selection")
            if frozen is None:
                raise ValueError("freeze the training selection before held-out evaluation")
            if self.config.experiment.stages:
                self._activate(self.config.experiment.stages[-1])
                results = {
                    name: self._step(
                        "postrun:" + name,
                        endpoint,
                        lambda endpoint=endpoint: self.research.postrun(endpoint["candidate"]),
                        repeatable=True,
                    )
                    for name, endpoint in frozen["stage_selections"].items()
                }
                return self._step(
                    "postrun", frozen, lambda: {"stage_endpoints": results}, repeatable=True
                )
            return self._step(
                "postrun",
                frozen,
                lambda: self.research.postrun(frozen["candidate"]),
                repeatable=True,
            )
