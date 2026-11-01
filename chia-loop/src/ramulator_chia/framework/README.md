# CHIA campaign harness

`ramulator_chia` provides the DRAM experiments, metrics, model selection and the
agents' workspace. The pinned `chia` package provides the LLM sessions,
transport, scheduling and persistence. The commands are in the root
`REPRODUCING.md`.

## How a campaign runs

A campaign runs 10 single-core iterations, then 5 iterations that add four- and
eight-core mixes. In each iteration the agent develops on named training
workloads and submits one candidate. The harness measures it on the training and
anonymous validation workloads, and a reviewer of the same model and effort
decides whether it replaces the current model. The reviewer may accept a
justified trade-off; mechanical integrity checks must always pass. The test
workloads stay hidden until the final models are selected.

The prompts are in `prompts/staged_v1/`. The campaign records the model selected
at the end of each stage and the iteration that introduced it.

## Diagnostics, recovery and isolation

The optional synthetic and open-loop tools stop a simulation after 1,800 seconds
and report the run as failed, keeping its evidence. Full evaluations use the
limits in the campaign configuration. Time the agent spends waiting for a result
does not count against its session.

The harness saves native events, tool effects, summaries, decisions and usage
records as it goes, so an interrupted campaign resumes where it stopped. A
failure it cannot classify stops the campaign for inspection.

Each campaign gets its own workspace, sandboxed with Landlock and seccomp. The
agent sees only what the workspace allows: `results/`, credentials, other
campaigns and test evidence stay outside it.

`scripts/run-campaign` needs `--allow-paid`, a configuration, a runtime and a
tariff; it is the only command that calls an LLM. The bundled tariffs hold the
September 2026 prices used for the cost estimates. `upstream/README.md`
describes the changes to CHIA.
