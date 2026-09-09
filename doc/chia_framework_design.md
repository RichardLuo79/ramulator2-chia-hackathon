# A small CHIA loop for atomic DRAM research

The common workflow for Gemini, Astra and Fable. See the
[implementation status](chia_framework_implementation.md) for current evidence
and remaining limitations.

## Goal and workflow

Develop a generic immediate-response DRAM model using agents, measure its
accuracy and speed, and retain evidence that explains the outcome. Building a
general experiment platform is not a goal. Use one workflow for all backends.

```text
 campaign.json       task + seed + approved references       workload manifest
       |                           |                       train / test / transfer
       +---------------------------+------------------------------+
                                   |
                   validate inputs; build with -O3
                   prepare training oracle and baselines
                                   |
                 +-----------------v--------------------+
 prior summary   | CHIA iteration                       |
 + own history ->| continuous native agent session      |
                 |   inspect / edit / build             |
                 |   training evaluation + diagnostics  |
                 |   optional independent critique      |
                 |   revision in same proposer session  |
                 |                                      |
                 | final candidate -> checks + training |
                 | promotion decision -> model summary  |
                 +-----------------+--------------------+
                                   | repeat to run guard
                                   v
                          freeze selected source
                                   |
                held-out SimpleO3 + ChampSim/gem5 DDR5 transfer
                                   |
                results / plots / compressed source-and-data archive
                                   |
                independent asynchronous review and analysis
```

Each stage records evidence in one campaign directory. A small checkpoint
database records progress. Testing requires frozen source and configuration,
not a completed archive. Packaging never launches missing evaluations.

## Configuration and visibility

| Group | Contents |
| --- | --- |
| Experiment | DDR5 setup, workload manifest and splits, diagnostic switches, semantic-review boolean |
| Backend | CHIA backend, exact model, effort and necessary provider-specific settings |
| Run | Iterations, CPU allocation, practical timeouts and bounded retries |

Accept only implemented settings with tests showing their effect. Preserve
effective settings, source/input identities and build commands. Credentials
stay outside configuration and artifacts. Splits are fixed before search,
separated by application family, with duplicate trace hashes rejected across
splits. Initial training is SimpleO3; frontend transfer is post-search only.

Build the agent view from an explicit allowlist: task, model API, seed/current
source, approved references, its own campaign history, and enabled training
tools/results. Do not mount the repository, Git history, operator home, other
campaigns, held-out inputs or independent feasibility design. Enforce this with
the filesystem/tool boundary, not just a prompt. Approved reference visibility
is a declared choice, applied equally across backends.

Keep one native proposer session throughout each iteration. Preserve its native
continuation, including opaque reasoning/tool-call state; do not reconstruct
history from final answer text. Record exposed native events without claiming
access to private reasoning the provider does not expose. After evaluation the
model writes `summary.md`: changes, reasons, results, failures and next questions.
The next iteration receives that summary and approved same-campaign history.
A new campaign starts clean.

## Scientific contract

- The model uses the existing immediate-response API: predict departure at
  admission, without a cycle-level scheduler, future requests, oracle answers
  or workload-specific answers. Model-specific parameters are editable;
  evaluator configuration and observations are not.
- Scientific runs use `-O3`, at most 12 evaluation CPU cores in total, and at
  least 20M instructions/core for SimpleO3, with fixed issue ROI, no trace wrap
  and full drain. Tiny fixtures qualify protocol behavior, not accuracy.
- Objective 1: mean absolute per-core cycle percentage error within each
  workload, then an equal-weight mean across workloads.
- Objective 2: paired request latency MAE divided by that workload's oracle
  mean latency **L**, then an equal-weight mean across workloads. Preserve the
  existing observation populations, timebases, pairing and traffic checks.
- Promote a mechanically valid final candidate only when neither training
  objective worsens and at least one improves. Use unrounded values. Missing
  or invalid workloads cannot improve the mean. Multiple diagnostic drafts
  are allowed, but each iteration submits one final candidate.
- Reuse completed evaluations only when source, parameters, inputs and
  evaluation settings match. Test and transfer scores never drive search.
- Keep the cycle-level oracle and FixedLatency, M/D/1, published channel-level
  WMG1 and MeSS comparisons. Exclude the unpublished bank-level Anatomy model.
- Report per-workload scores, signed drift, tails/extremes, coverage and runtime
  beside the headline objectives. If transfer request identity is incomplete,
  withhold its whole-population request score; retain valid core-cycle results.
- Generic synthetic patterns and open-loop replay are selectable training
  diagnostics, not promotion datasets. Import no findings or diagnostic cases
  from the independent non-CHIA exploration.

`semantic_llm_check` is a boolean. When enabled, one independent session of the
same model and effort gives one critique against the rules. It receives no
hidden data or prior design. The proposer may revise in its original session.
The critique is advisory, not a promotion veto or repair loop. Record its usage.
Separate post-campaign review can examine the frozen source, results and logs;
it cannot change the completed search.

## Evidence, recovery and artifacts

Retain prompts, exposed native events, tool calls/results, candidate sources,
summaries, build logs, raw observations, scores and usage. Separate estimated
API-equivalent cost from subscription billing. Keep native counter scope;
unknown usage is not zero. Do not demand per-request counts from a CLI that
only exposes totals.

Checkpoint completed steps. Retry recoverable failures within recorded bounds,
without silently repeating a possibly charged operation or resetting accounting.
Keep partial evidence and report unresolved failures. Do not build a generic
workflow recovery engine or reconstruct arbitrary lost working directories.

Compress large traces/logs as they close and validate compressed inputs before
use. Reuse immutable verified inputs rather than decoding an entire trace for
each small diagnostic page. Preserve originals until compression is verified.

The final compressed, checksummed artifact contains our source, configuration,
inputs, candidates, metadata, logs and results. External projects use upstream
URLs, immutable revisions, submodule pins, checksummed patches and build recipes.
Do not bundle executables, shared libraries, wheels, installed environments or
container images. Binary-format data is still data. Rebuilding may fetch
dependencies; offline environment restoration is not a requirement.

## Reuse, implementation order and completion

CHIA owns native sessions, tool transport, scheduling and database primitives.
Our layer owns DRAM tools, scientific policy, positive visibility and evidence.
Reuse existing instrumentation, generators and metric calculations. Keep only
necessary, documented CHIA extensions. No second scheduler, provider protocol,
package manager or speculative configuration modes.

1. Prove an executable common iteration using scripted model responses:
   training tools, optional critique, promotion, summary, next iteration, resume.
2. Attach existing DRAM evaluators and CHIA native adapters to that workflow;
   test continuation and positive visibility offline.
3. Exercise full-window comparisons and frozen DDR5 transfer; report genuine
   scientific limitations instead of hiding them.
4. Export source/evidence, check archive integrity, and document operator commands.

Preserve useful scientific regressions; retire tests for removed features.
Paid inference requires explicit authorization. The current common launcher
uses an iteration guard and records cost estimates; it does not implement a
dollar spending cap. Docker deployment is **[OPTIONAL]**, after the loop
converges, using CHIA workers and source builds without image export.

Done means the common loop runs, supported settings are exercised, and relevant
offline evidence is recorded. Test counts and infrastructure features do not
substitute for an executable research workflow.
