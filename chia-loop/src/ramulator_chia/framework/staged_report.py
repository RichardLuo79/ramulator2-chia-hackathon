"""Read a staged campaign's records into compact JSON and Markdown. No inference."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from .config import CampaignConfig
from .identity import canonical_json
from .review import score_view
from .snapshots import publish_bytes


def usage_summary(root: Path) -> dict:
    # Terminal native receipts already select the right accounting scope. Do
    # not also add their nested HTTP events or copies in campaign checkpoints.
    attempts, incomplete, rows = 0, 0, []
    for path in sorted((root / "native-evidence").glob("*/*/*/receipt.json")):
        receipt = json.loads(path.read_text())
        if "usage" not in receipt:
            continue
        attempts += 1
        rows.extend(receipt["usage"])
    for path in (root / "native-evidence").glob("*/*/*/prompt.txt"):
        if not path.with_name("receipt.json").exists():
            incomplete += 1
    result = {
        "native_attempts": attempts,
        "receipts": len(rows),
        "attempts_without_usage_receipt": incomplete,
        "cost_basis": "estimated API equivalent, not subscription spending or invoice",
    }
    for key in ("lower_micro_usd", "upper_micro_usd"):
        values = [row.get("cost", {}).get(key) for row in rows]
        result[key] = (
            sum(values)
            if values and all(v is not None for v in values) and not incomplete
            else None
        )
        result["known_" + key] = sum(v for v in values if v is not None)
    result["tokens"] = {}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens"):
        values = [((row.get("usage") or {}).get("tokens") or {}).get(key) for row in rows]
        result["tokens"][key] = (
            sum(values)
            if values and all(v is not None for v in values) and not incomplete
            else None
        )
    return result


def collect(root: Path) -> dict:
    with sqlite3.connect(
        (root / "campaign.sqlite").resolve().as_uri() + "?mode=ro", uri=True
    ) as database:
        records = {
            key: json.loads(value)
            for key, value in database.execute("SELECT key,value FROM records")
        }
        postrun = database.execute(
            "SELECT result FROM attempts WHERE step='postrun' AND status='complete' "
            "ORDER BY number DESC LIMIT 1"
        ).fetchone()
    configuration = CampaignConfig.model_validate(records["configuration"])
    if not configuration.experiment.stages:
        raise ValueError("this report is for staged campaigns")
    experiment = configuration.experiment
    rounds = []
    for number in range(1, configuration.run.maximum_iterations + 1):
        row = records.get(f"iteration:{number}")
        if row is None:
            continue
        stage = experiment.stage_for(number)
        rounds.append(
            {
                "round": number,
                "stage": row["stage"],
                "candidate": row["candidate"],
                "selected": row["selected"]["candidate"],
                "promoted": row["promoted"],
                "decision": row["promotion_review"],
                "mechanical_reason": row["selection_reason"],
                "training": score_view(
                    row["training"] or {}, experiment.case_names("training", stage), anonymous=False
                ),
                "validation": row["validation"],
                "summary": row["summary"],
            }
        )
    return {
        "schema_version": 1,
        "campaign_id": configuration.campaign_id,
        "execution": configuration.execution,
        "backend": configuration.backend.model_dump(mode="json"),
        "completed_rounds": len(rounds),
        "planned_rounds": configuration.run.maximum_iterations,
        "stages": [stage.model_dump(mode="json") for stage in experiment.stages],
        "stage_endpoints": {
            s.name: records.get("stage-selection:" + s.name) for s in experiment.stages
        },
        "rounds": rounds,
        "postrun": json.loads(postrun[0]) if postrun else None,
        "usage": usage_summary(root),
        "test_status": (
            "Historical cohort previously examined by the research team; "
            "hidden during all search rounds."
        ),
    }


def markdown(report: dict) -> str:
    def number(value):
        return "unavailable" if value is None else f"{value:.5g}"

    lines = [
        f"# {report['campaign_id']}",
        "",
        f"Completed rounds: {report['completed_rounds']}/{report['planned_rounds']}. "
        f"Execution: `{report['execution']}`.",
        (
            "Backend: scripted fixture responses, not model development."
            if report["backend"]["kind"] == "fixture"
            else f"Backend: `{report['backend']['model']}`, "
            f"effort `{report['backend']['reasoning_effort']}`."
        ),
        "",
        "Groups have equal case weights and are never pooled across core counts. "
        "The tables track submitted candidates; selection follows the recorded review.",
        "",
        "| Round | Stage | Decision | Split | Cores | Core MAE (%) "
        "| Request MAE/L | P99/L | P99.9/L |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rounds"]:
        for split in ("training", "validation"):
            for cores, group in row[split]["groups"].items():
                mean, request = group["means"], group["means"]["request"] or {}
                verdict = "promote" if row["promoted"] else "keep"
                lines.append(
                    f"| {row['round']} | {row['stage']} | {verdict} | {split} | {cores} | "
                    + " | ".join(
                        number(v)
                        for v in (
                            mean["core_error_pct"],
                            request.get("mae"),
                            request.get("tail"),
                            request.get("p999_over_L"),
                        )
                    )
                    + " |"
                )
    lines += ["", "## Selection record", ""]
    for row in report["rounds"]:
        decision = row["decision"] or {}
        lines += [
            f"### Round {row['round']}",
            "",
            decision.get("rationale", row["mechanical_reason"] or "No review recorded."),
            "",
        ]
        for key in ("improvements", "accepted_regressions", "contract_findings", "uncertainty"):
            if decision.get(key):
                lines += [key.replace("_", " ").capitalize() + ": " + "; ".join(decision[key]), ""]
    lines += ["## Frozen endpoints and held-out evaluation", ""]
    for stage, endpoint in report["stage_endpoints"].items():
        lines.append(
            f"- {stage}: "
            + (
                f"round {endpoint['iteration']}, `{endpoint['candidate']['candidate_id']}`"
                if endpoint
                else "not frozen"
            )
        )
    lines += [
        "",
        report["test_status"],
        "",
        "Held-out tables are in the companion JSON."
        if report["postrun"]
        else "Held-out evaluation has not completed.",
        "",
        "## Usage",
        "",
        "```json",
        json.dumps(report["usage"], indent=2),
        "```",
        "",
        "The companion JSON retains per-case/per-core metrics, tails, coverage, "
        "endpoint receipts and review decisions. Missing data are not zeros. "
        "Native logs, source identities and summaries remain in the campaign archive.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="new Markdown report; JSON is written alongside"
    )
    args = parser.parse_args()
    report = collect(args.campaign)
    publish_bytes(args.output, markdown(report).encode())
    publish_bytes(args.output.with_suffix(".json"), canonical_json(report).encode())


if __name__ == "__main__":
    main()
