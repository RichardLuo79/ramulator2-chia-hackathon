# Unattended CHIA execution

Current Gemini protocol: `gemini_individual_run_v11`. Effective scientific and
operational limits are frozen in each run. This document describes recovery
and compliance; the [project snapshot](chia_hackathon_review.md) records current
results. Historical protocols and their original evidence remain unchanged.

## Compliance without interactive supervision

Compliance has two layers. Deterministic checks enforce the editable boundary,
permitted interfaces, successful optimized compilation, isolated execution,
drained callbacks, and exact request pairing. A separate API reviewer examines
the properties that these checks do not establish for arbitrary C++: bounded
causal state, immutable predictions, absence of a command scheduler, generic
configuration, prohibited access, and technically explained approximations.

The reviewer is an automated service, not a person or this interactive coding
session. Both proposing backends use Gemini 3.1 Pro Preview with HIGH thinking
and the same versioned [rubric](../tools/chia_loop/prompts/compliance_v1.md).
It receives only the candidate source and technical explanation, in a separate
conversation with no tools, scores, search history, or other-run information.
Its decision identifies the source, review input, rubric, and reviewer model.

Every rule must pass. Reject or uncertain decisions return compliance feedback
to the proposer, which alone can change the source. Invalid review responses
receive up to three formatting attempts and otherwise become uncertain. The
reviewer is instructed not to supply implementations or tuning values and does
not control numerical promotion. Its free-text reasons are not mechanically
proven advice-free; all returned feedback is retained for retrospective review.
Transport failures follow recovery rather than becoming negative model reviews.

This design needs no routine human approvals. It does not claim that LLM review
is a formal proof. Removing semantic review entirely would require a different
assurance strategy, such as restricting candidates to a verifiable modeling
language; lexical checks and finite workload tests alone are insufficient for
the current general C++ interface.

## Operational recovery and scientific termination

Each paid attempt is reserved and journaled before dispatch. Proposer and
reviewer calls have distinct roles and tariffs but share one run's cap. Thinking
tokens count as output. A lost response keeps its conservative reservation;
retrying requires another affordable reservation. A saved complete response is
replayed without another generation. These are conservative estimates, not
Cloud Billing invoices.
The per-iteration limit of 60 generation attempts includes both roles and their
retries. Nonbillable token-count requests are recorded operationally but are
not generation attempts.

Checkpoints retain the active parent, exact prompt/conversation and returned
signatures, diagnostic/draft counters, candidate lineage, and selected state.
Recovery cannot substitute another parent or reset the budget. Build and
evaluation caches verify source/binary/configuration/input/trace identities.
Abandoned simulation artifacts remain separately ineligible for scoring.

Transient connection and token-count failures, including unexpected HTTP
`499 CANCELLED` responses, get up to two immediate retries
with increasing delays and jitter, followed by a persisted cooldown. New
connections are used for new attempts. The supervisor resumes the same operation
and iteration after cooldown; it does not count an outage as an evaluated design.
Recovery guards are eight transient failures per operation, a one-hour outage
window, a 24-hour run window, and three abrupt-process restarts. In-flight work
can take its bounded completion time; STOP is checked before dispatch, before
replaying saved responses, and during retry waits. An explicit STOP always
prevents further dispatch; it is never treated as a retryable provider error.
A `499` status alone does not establish who initiated the cancellation or prove
a provider outage. Retries retain the same operation and request identity,
and consume the existing failure, generation-attempt and spending allowances.

Infrastructure failures never trigger final testing. Authentication/configuration
faults, corrupted evidence, insufficient disk space, and persistent outages
stop safely and report that attention is needed. No automatic deletion, protocol
change, model substitution, or spending-cap increase is permitted.

Selection ends only at the configured design limit, an explicit no-change
decision, or a declared budget/search guard. The incumbent then freezes before
held-out evaluation. A restart after this boundary may finish that frozen
evaluation or archive verification, never resume optimization. Old test-exposed
experiments cannot be upgraded into this protocol.

### Recorded v8 to v9 recovery amendment

The September 5 runs were safely paused to add the narrow `499` classification
and entry STOP check. The operator utility
`tools/chia_loop/maintenance/amend_retry499.py` captures original manifests,
ledgers, scientific state, checkpoints and affected frozen source before the
change. Applying it requires both worker locks, an explicit maintenance STOP,
an unfinished search with no test exposure, and exact validation that no other
protocol or scientific settings changed. It does not launch a model or remove
the STOP marker.

Only a blocked checkpoint matched to its original `499` ledger receipt can be
rearmed. The previously uncounted failure consumes one of the existing eight
failure slots. A new one-hour recovery window starts at explicit activation
only if none existed; an existing window is never reset. Successful responses
remain replayable without another paid generation. The original evidence and
amendment receipt remain under `operational_amendments/` in each affected run.
Both runs record a human operational intervention and no human modeling hints;
their previous scientific results and financial history are unchanged.

## Resources, records, and use

Current fresh Gemini rich-DDR5 campaigns were configured independently for
at most 20 evaluated designs and USD 100 each, including reviewer calls and
retries. Generic defaults remain five designs and USD 50 unless preparation
explicitly pins larger limits. An authorized financial continuation retains
earlier charges; a new directory never implicitly replenishes a budget.

Evaluation remains full-window SimpleO3 + DDR5: at least 20 million issued
instructions per core, complete drain, and explicit `-O3`. Training/test isolation,
both primary errors, Pareto promotion, and FixedLat/MD1/WMG1/MESS comparisons are
unchanged. Current campaigns request three CPU slots each; a shared local lease
pool prevents runners from claiming more than twelve slots. Preparation and
unrelated evaluation jobs must respect the same overall allowance.

Traces are gzip-compressed and verified after each simulation. Closed interaction
records are compressed between iterations; live checkpoints remain directly
readable. Recovery and diagnostics read archived interactions transparently.
Finalized logs/profiles are compressed after worker shutdown. An 8 GiB free-space
floor prevents starting additional generation/evaluation work on a nearly full
disk; it is not a guarantee against another process filling the disk.

After preparing and authorizing an independent run:

```sh
.venv/bin/python -m tools.chia_loop.supervise_gemini --root eval_out/chia/RUN_ID --model pro
```

Use `--model flash` and a different root for Flash. To reattach to that same
protocol's unfinished run, add `--resume`. The runner verifies actual frozen
bytes and dependency versions before resuming. Create `RUN_ID/STOP` to stop
safely. The supervisor records launch/recovery status and per-launch logs;
the runner records all API attempts, review decisions, repairs and evaluations.
It never commits or pushes.

Automated tests inject disconnects, crashes, malformed reviews, cap exhaustion,
and final-test markers without calling a provider. The opt-in real integration
test uses `-O3` and a full 20M-instruction case to verify completed build/trace
reuse. These tests validate infrastructure, not Gemini's modeling performance
or the quality of live semantic reviews.
