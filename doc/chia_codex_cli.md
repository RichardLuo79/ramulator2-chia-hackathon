# Isolated Codex CLI backend for CHIA

This describes the historical runner. New campaigns use the
[common framework](../tools/chia_loop/framework/README.md), including its native
session continuity and advisory-review policy. Limits and queue commands below
apply only to the legacy protocol.

This backend runs independent GPT-6 Astra xhigh/max experiments through the
unmodified native CLI with the user's existing subscription login. Current
fresh rich-DDR5 campaigns use 20-iteration guards, the eight-family training
and eight-family final-test profile, generic synthetic diagnostics, and
post-freeze ChampSim/gem5 transfer. See the
[current project snapshot](chia_hackathon_review.md) for results and limitations.

Preparation and generation are separate gates. No prior campaign, feasibility
design, native CLI session, or operator conversation enters a fresh run.
An explicitly authorized continuation is a different experiment: it imports
only its own completed training checkpoints, must reproduce their training
scores, and preserves financial history. Its repeated test is an exploratory
comparison on a reused holdout, not a new untouched test.

## Iteration-guarded continuations

`--iteration-guard` is mutually exclusive with `--usd-cap` and is available only
with the explicitly configured ChatGPT login. It records `guard_mode=iterations`
and `usd_cap=null`; it is not an infinite or fabricated dollar allowance.
Token receipts, tariff-equivalent usage, conservative reservations for unknown
outcomes, and predecessor usage remain recorded. Provider quotas, STOP, finite
turn/draft/retry bounds, the 24-hour operational guard, disk guards, compliance,
process isolation, and the shared twelve-CPU lease pool remain enforced.

Use `--continue-training-from` only for a completed same-model, same-effort
predecessor. The importer preserves all evaluated candidates, training history,
the incumbent, and therefore the same Pareto parent pool. It recreates the
protected runtime and comparisons, rebuilds the inherited candidates with
`-O3`, and requires exact reproduction of training accuracy, populations and
coverage; measured host runtime is not required to repeat. Source-bound
automatic compliance decisions are retained, not replaced by human approvals.
An incomplete budget-stopped proposal remains archived in its original run;
continuation resumes at the last completed training checkpoint.

No predecessor test artifacts, other effort's history, feasibility design,
or orchestrator conversation enter the agent input. Earlier interactions stay
in their original compressed archives, linked by `continuation.json` and the
retrospective report. The new run's final selection freezes before its test
replay. Because the operator already saw the earlier test scores, that replay
must be described as an exploratory comparison on a reused holdout.

For example, explicitly authorize one queued continuation:

```sh
python -m tools.chia_loop.codex_cli.queue \
  --root eval_out/chia/ASTRA_XHIGH_CONTINUATION --effort xhigh \
  --after eval_out/chia/ASTRA_XHIGH_COMPLETED \
  --continue-training-from eval_out/chia/ASTRA_XHIGH_COMPLETED \
  --auth-file /home/dev/.codex/auth.json \
  --max-iterations 20 --iteration-guard --cpus 6 --authorize-paid
```

Use a separate invocation and same-effort predecessor for `max`. Ordinary
`--resume` still cannot reopen a frozen run or erase test-exposure markers.
Continuation copies only approved scientific inputs and retains full financial
lineage; it never resumes a native Codex conversation.

## Experiment

Two **independent runs** use the exact model `gpt-6-astra`, one with `xhigh`
reasoning and the other with `max`. Both values are advertised by the installed
Codex CLI 0.153.4 catalog and the [Astra model documentation](https://developers.openai.com/api/docs/models/gpt-6-astra).
Neither a model substitution nor a reasoning-effort downgrade is permitted.
Catalog support does not prove that a particular account can make a paid call.

Both efforts and the common reviewer explicitly request readable reasoning
summaries using `model_reasoning_summary="auto"` and
`model_supports_reasoning_summaries=true`. The broker checks the actual CLI
request's `reasoning.summary` before auth, reservation, or dispatch; missing or
different values stop the operation. Real-CLI, fake-provider tests verify
both outgoing requests and preservation of returned summaries, including
streams whose terminal response omits the already-streamed output items.

Each initial run starts from the unchanged fixed-delay Atomic skeleton; an
explicit continuation inherits only its own completed training checkpoints.
Training uses SimpleO3 + DDR5 and the run's frozen evaluation profile, with
the established 20M-instruction cold prefix, full drain, and explicit `-O3`.
Legacy runs retain their original mcf/lbm training cohort. Held-out families, callback/trace instrumentation,
exact request matching, error definitions, and the FixedLat/MD1/WMG1/MESS
comparisons are reused from the protected evaluator. The under-review bank-level
Anatomy model and the earlier feasibility design are not exposed.

Promotion requires strict Pareto improvement of core-cycle macro absolute
percentage error and request macro MAE/L. Incomparable candidates can remain
parents in the nondominated archive. Selection freezes before held-out testing;
an infrastructure failure does not open the test split or consume an evaluated
design iteration. Rejected drafts are repaired by the proposing model only.

A source-bound automatic compliance reviewer uses Astra at **fixed `xhigh` for
both proposing efforts**, with the existing six-rule rubric. Its isolated input
contains only that source and its technical explanation, not accuracy scores,
search history, another run, or operator discussion. Uncertain/reject decisions
do not pass. Review is an additional semantic check, not a correctness proof.

The two Astra runs therefore share the same reviewer. A later comparison with
Gemini is an end-to-end system comparison, not a perfectly controlled comparison
of bare models: the CLI wrapper and reviewer service differ. The CLI adapter
reconstructs its own explicit text history for every action; it does not resume
a native Codex conversation or preserve an opaque remote reasoning state.

## Separation from the orchestrator

The orchestrator and the experimental agent can use the same model weights
without sharing an inference context. The important controls are what enters
each request and what the experimental process can read or call:

1. Every CHIA action launches a fresh `codex exec --ephemeral` process, never
   `resume`, `fork`, `queue`, a collaboration subagent, or an existing app server.
   Its only experiment input is the frozen contract and its own explicit CHIA
   conversation. It gets a private per-invocation Codex home and a scrubbed
   environment; the orchestrator's home/environment are not changed.
2. `--ignore-user-config`, `--ignore-rules`, disabled memories, plugins, apps,
   hooks, native tools, and instruction discovery prevent implicit context
   import. Private host instruction-file and environment canaries are checked
   against the actual CLI-generated outgoing request in the integration tests.
   See [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
   and [configuration controls](https://learn.chatgpt.com/docs/config-file/config-reference).
3. A whole-process Landlock/seccomp boundary is installed **before Codex starts**.
   Read-only CLI mode alone is not the privacy boundary. The outer boundary
   grants only system runtime files, the pinned CLI executable, and that one
   invocation's workspace. It does not grant the repository, `.git`, other run
   directories, host home, session/memory databases, credentials, or `/proc`.
   Tests exercise absolute paths, `..`, escaping symlinks, and host auth paths.
4. Unix sockets, UDP, non-broker TCP ports, cross-domain signals and process
   inspection are restricted. Landlock's TCP filter is **port-based**, not an
   IP-address firewall. The pinned CLI is configured to use only the loopback
   broker; the broker implements one authenticated Responses operation, not a
   generic HTTP proxy. This assumes a trusted CLI/runtime and kernel, not an
   adversarial replacement executable or hostile host administrator.
5. Native provider tools are removed at the broker boundary and native-tool
   responses are rejected before reaching Codex. The model requests source or
   diagnostic information using the same JSON actions as the Gemini adapter.
   CHIA serves only frozen allowlisted source and training observations. It
   assembles proposed region bodies and runs builds/simulations outside the CLI
   boundary using the existing separately restricted compiler/runtime.
6. The trusted broker owns credentials and makes only the configured provider
   Responses request. It rejects model/effort substitution and remote
   conversation identifiers. Provider requests, returned responses, usage,
   CLI events, source/review bindings and hashes are recorded outside the
   experimental process's writable domain. No API key or account token is
   passed into the CLI process or logged in its request files.

These controls prevent deliberate import of this chat, prior branches and
prior designs. They do not claim that a pretrained model has never encountered
related public work, or that arbitrary C++ and a trusted computing base have a
formal noninterference proof. Changing the experimental prompt using insights
from the feasibility design would still be an intervention, even with perfect
process isolation; the operator must not do that for a clean-start trial.

## Accounting, recovery, and authentication

Every broker dispatch reserves a conservative amount before calling the
provider. A lost response retains that reservation. Complete persisted
Responses are replayed and settled idempotently after interruption; the runner
does not pay again merely because the CLI failed while printing the answer.
There are at most three transport attempts per action, with cooldown, and a
separate bounded process supervisor. Authentication and integrity failures stop
for attention; they do not prompt the experimental model for credentials or
waive checks. A 24-hour run guard and the shared twelve-CPU lease pool apply.

The selected mode for this study is the operator's **existing ChatGPT/Codex
login**, not an API key. The trusted broker reads only the specified
`auth.json`; it never copies the host Codex directory. Before a dispatch,
credentials close to expiry are renewed by a short-lived trusted **auth-only**
Codex app-server helper. Its only requests are `initialize` and
`account/read`, with the `initialized` notification. It does not start, resume,
list or read a thread, and never calls a generation endpoint. Codex owns the
OAuth refresh flow and writes its existing credential store; CHIA does not
implement a competing refresh-token flow. See the
[managed-auth API](https://learn.chatgpt.com/docs/app-server#authentication-endpoints).
Authentication failures occur before reserving generation usage. Revoked
credentials or failed renewal stop safely for attention, with no model-supplied
credential handling or API-key fallback.

The installed CLI's non-generation auth-only check recognized the existing
ChatGPT login, and subsequent live requests confirmed model access through
the broker. Expiry/refresh and credential-boundary behavior also have offline
tests. Account recognition alone is not a generation test. No tokens should
be pasted into chat or committed. In this mode, a USD
figure is an **API-equivalent experiment guard**, not a monetary subscription
cap; ChatGPT/Codex quotas and any account-level usage policy remain separate.

An alternative API-key mode is implemented but is not selected for this study.
In that mode, the trusted runner reads `CHIA_OPENAI_API_KEY`. It never
passes that variable to Codex or candidate code. The broker requests at most
128,000 output tokens, disables inherited priority routing and hosted tools,
and charges both proposing and review calls to the same run. Accounting uses
conservative long-context input/cache-write and output rates for the guard;
known ordinary API-cost estimates are also reported. These are estimates, not
Cloud Billing or OpenAI invoices.

Workload traces use the existing verified gzip workflow. Large CLI JSON/log
artifacts and provider SSE receipts are compressed after execution. Replay
reads either raw or gzip-compressed evidence and verifies the response hash;
compression must not cause another generation charge. Free-space guards remain
active throughout the run.

### Finalized streamed output and clean-restart accounting

The first live v3 trials returned successful provider responses and valid JSON
actions in `response.output_item.done` events, while the final
`response.completed.response.output` array was empty. The old adapter inspected
only that final array and incorrectly supplied invalid-response feedback.
Xhigh exhausted 48 calls in its first iteration; max was safely stopped after
47 calls. Neither evaluated a proposed controller. Historical feedback,
ledgers, raw streams and any seed-only held-out results are preserved as-is.

V4 reconstructs finalized items in output-index order and reconciles them with
any embedded terminal output. It requires a matching terminal lifecycle event,
rejects contradictory/missing/unfinished items and native tool output, and does
not treat partial text deltas as a complete answer. A completed stream without
a finalized answer now stops as an infrastructure fault rather than consuming
the remaining model turns as invalid-response feedback. Raw bytes, hashes,
usage accounting, exposed summaries and gzip replay are preserved. This follows
the Responses API's separation of output events from completion events;
see [streaming responses](https://developers.openai.com/api/docs/guides/streaming-responses).

The fix recovered the JSON actions from all 95 captured trial responses without
another provider call. Synthetic tests cover both populated and empty terminal
arrays, malformed/incomplete streams and forbidden tool items. Real-CLI tests
exercise both response forms at both efforts, including saved and compressed
replay. The corrected trials start from the unchanged skeleton and fresh
training preparation, not from recovered controller proposals or conversations.

For an explicitly authorized clean restart, preparation and queue commands
accept `--carry-budget-from PREVIOUS_SAME_EFFORT_ROOT`. The predecessor must be
quiescent with released locks; an unfinished predecessor must also retain STOP.
Only its financial ledger, configuration identity and supervisor status are
read. A hash-bound `budget_carryover.json` carries total known usage and all
conservative charges, including unknown-outcome reservations. Prior source,
prompts, scores, histories and held-out outcomes are never imported. A
predecessor can have only one financial successor, preventing reuse of its
remaining authorization in two fresh runs.

The ceiling remains USD 100 including carryover, not USD 100 of new allowance.
The September 5 replacements carry $23.576775 for xhigh and $24.139500 for max
against that guard; corresponding known API-equivalent usage is $8.521082 and
$9.066764. Current-run call IDs, role/iteration reports and token totals remain
separate. `reports/llm_usage/summary.json` exposes both current-run `totals` and
combined `authorization_totals`; the prompt receives financial totals only.
Operational repair/restart provenance is documented outside the agent context.

## Commands and remaining launch checks

Preparation is nonbillable. `--wait-for-cpus` queues it for free slots in the
shared twelve-CPU pool. Without that flag, occupied slots cause a clean pause
before the run directory is created. Each effort must use a distinct root.
The first preparations use 10 evaluated iterations, with a provisional $100
API-equivalent guard per run. This guard is not permission to consume account
quota: model execution still requires a separate explicit launch.

```sh
python -m tools.chia_loop.codex_cli prepare \
  --root eval_out/chia/ASTRA_XHIGH_RUN --effort xhigh --auth-mode chatgpt \
  --auth-file /home/dev/.codex/auth.json \
  --max-iterations 10 --usd-cap 100 --cpus 6 --wait-for-cpus
```

For the other independent run, choose another root and `--effort max`.
ChatGPT mode is the default; `--auth-file` remains explicit to avoid accidental
credential-store selection. That path is operator-only, not model input.
To resume the same pinned run, use `--resume` with `run` or `supervise`.
Create `RUN_ROOT/STOP` to request a safe stop between operations.
For queued preparation, the sibling `RUN_ROOT.preparation_status.json` records
waiting/preparing/prepared/failure state. Create the sibling
`RUN_ROOT.preparation_STOP` to cancel its wait. Preparation never chains into
generation. This sibling marker also blocks direct preparation, `run`,
`supervise`, and generation dispatch; it is not bypassed by `--authorize-paid`.
Do not remove an operator hold until its recorded readiness issue is resolved.
After launch is authorized, use:

```sh
python -m tools.chia_loop.codex_cli supervise \
  --root eval_out/chia/ASTRA_XHIGH_RUN --authorize-paid
```

Before any usage-consuming study: confirm the new Astra authorization,
complete full-window preparation when CPU slots are available, and verify the
chosen account's live transport/model access. The first paid action can be
that check; no separate paid diagnostic call is required. The USD 100 Gemini
authorizations do not authorize additional Astra spending.

Offline tests use the real CLI and kernel boundary with a local fake provider,
plus injected budget, replay, rejection, review, promotion and freeze cases.
They do not demonstrate paid Astra model quality, live token renewal,
or completed Astra experiments.

### Dependency-ordered unattended queue

`tools.chia_loop.codex_cli.queue` schedules one independently authorized effort:

```sh
python -m tools.chia_loop.codex_cli.queue \
  --root eval_out/chia/ASTRA_XHIGH_FRESH_RUN --effort xhigh \
  --after eval_out/chia/GEMINI_PRO_RUN eval_out/chia/GEMINI_FLASH_RUN \
  --auth-file /home/dev/.codex/auth.json \
  --max-iterations 10 --usd-cap 100 --cpus 6 --authorize-paid
```

Use a separate invocation/root for `max`. The explicit authorization covers
preparation followed by model execution through the existing ChatGPT login;
no separate diagnostic generation is issued. The queue waits for **all**
predecessor supervisors to be `completed`, `needs_attention`, or `stopped`,
and for their supervisor/runner locks to be released. A predecessor failure
counts as terminated, not successful, and is never repaired by this queue.
It reads only supervisor status and locks, not predecessor source, prompts,
metrics or histories. Queue dependency records are operator-only siblings of
the fresh run directory and never enter experimental prompts.

Each queue pins the current implementation, relevant build inputs and Codex
binary; changes before or during preparation stop it before model dispatch.
Full-window preparation reruns the established `-O3` checks and uses the pinned
CLI binary for the offline request/logging tests. Only a successful preparation
and matching frozen manifest lead to the existing unattended supervisor. Shared
CPU leases limit the two six-CPU evaluations to twelve total. Waiting for
predecessors is bounded to 24 hours; normal preparation/run guards also apply.

The sibling `RUN_ROOT.queue.json` records queue and final status. During
`supervising`, use the run's `supervisor_state.json`, `state.json`, and
`ledger.json` for live execution/usage; the queue does not poll or steer the
optimizer. Create `RUN_ROOT.preparation_STOP` to cancel a waiting queue or block
later dispatch, or `RUN_ROOT/STOP` for the normal safe stop once it exists.
Queues do not restart themselves after process/machine loss, overwrite existing
receipts, or silently resume a partial preparation. The run supervisor retains
its separately bounded worker recovery. No chat wake-up or external upload is
implied by this local scheduling mechanism.

## LLM interaction and usage records

Every effort owns its own run directory and ledger. The raw records describe
**only that experiment**; this orchestrator's chat, setup discussions, prior
design, other runs and host Codex history are not imported. Recording and
publication are separate decisions: raw interactions remain private under
the ignored `eval_out/` directory. No automatic publication/upload is provided.

| Evidence | Per-run artifact | Meaning |
| --- | --- | --- |
| Experiment identity | `codex_config.json`, `cli_identity.json`, `run_manifest.json`, `loop_config.json` | Exact requested model/effort, CLI version and binary hash, Python/dependency versions, prompts, protocol/input hashes, resolved ablation features, limits and reviewer policy. |
| Synthetic diagnostics | `diagnostics/synthetic_calls/`, `diagnostics/synthetic/` | Optional training-only generic experiments, parent DSO identities, parameters, action/result receipts, statistics and verified gzip traces. Never promotion scores or imported prior-design cases. |
| Full model input | `interactions/OP/input.json`, `attempt_NNN/cli_request.json`, `provider_request.json` | The explicit system/conversation input and the actual CLI/provider request bodies, including tool feedback and the broker's native-tool removal. |
| Execution settings | `attempt_NNN/cli_command.json`, `boundary.json` | Exact CLI options, isolation policy and clean child environment. The disposable broker token is redacted from the command record. |
| Provider operation | `attempt_NNN/provider_exchange.json` | Client/provider request IDs, HTTP status, timestamps, elapsed time, first-byte time, received-byte hash/count, and safe exception class. No auth headers, cookies or HTTP error bodies. |
| Response and CLI receipt | `provider_response.sse`, `cli_events.jsonl`, `cli_stderr.log`, `receipt.json` | Full returned stream, including partial bytes on interruption; CLI events/exit, last operation stage, returned JSON and usage cross-check. A first SSE byte is not necessarily a visible answer token. |
| Usage accounting | `ledger.json`, `attempt_NNN/reservation.json` | One reservation per generation attempt, proposal/review role, iteration/turn/draft, raw provider usage, normalized token counts, requested/reported model and effort, terminal status and tariff-equivalent estimates. |
| Design evolution | `candidates/`, `state.json`, `events.jsonl`, `profiles/` | Proposals/explanations, inspections and feedback, rejected drafts, source hashes, source-bound reviews, builds, measurements, promotions, parent selection and CHIA execution profiles. |
| Readable retrospective | `reports/retrospective/index.json`, `index.md` | Per-iteration actions, stated rationale, exposed summaries (including partial streams), diagnostic-feedback references, drafts, reviews, outcomes and links to raw evidence. Private, deterministic rendering; no extra model calls. |
| Intervention/recovery evidence | `events.jsonl`, `supervisor_state.json`, `supervisor_logs/` | Launch/resume, cooldown/retry, observed stops and integrity failures. No manual modeling hints or approvals are permitted by this initial-run policy. |

The model's requested effort is checked against the **actual outgoing request**.
A returned model/effort mismatch stops execution; if the provider omits its
effort field, the report does not invent a server-side attestation. Likewise,
only reasoning summaries/content actually exposed by the provider/CLI can be
stored. The previous v2 configuration recorded raw streams but did **not**
explicitly request readable summaries; v3 fixes that omission before any Astra
model generation. `reasoning.encrypted_content` is opaque, not a readable
thinking transcript. Hidden internal reasoning is not available and is not
claimed as logged. See [OpenAI reasoning summaries](https://developers.openai.com/api/docs/guides/reasoning#reasoning-summaries).

The retrospective labels summaries as `returned`, `partial`, `not_returned`,
or `unavailable`. No returned text does not mean no reasoning occurred; it does
not trigger extra model calls or invented explanations. Reasoning token counts
measure usage, not recoverable thought content. A model's explanation and its
provider-generated summary are claims to examine against source changes and
measurements, not a complete internal trace or proof of causality. This logging
requirement concerns observable work and provider-exposed summaries, not access
to hidden chain of thought.

Every submitted draft already requires evidence, a hypothesis, its mechanism,
genericity/atomicity and complexity arguments, expected effects, and limitations.
The raw operation inputs preserve exactly which diagnostic feedback preceded
each action. Rejected drafts, repairs, and reviewer responses are retained, not
only the promoted designs. Retrospective rendering does not rewrite prompts,
reconstruct hidden thinking, generate post-hoc LLM explanations, or expose new
information to either experimental agent.

Token definitions follow the [Responses usage schema](https://developers.openai.com/api/reference/typescript/resources/beta/subresources/responses/methods/create):

- `input_tokens`: total input, including reported cached input/cache writes.
- `cached_input_tokens` and `cache_write_input_tokens`: reported input subsets.
- `output_tokens`: total output, **including reasoning**.
- `reasoning_output_tokens`: the reported reasoning subset of output.
- `non_reasoning_output_tokens = output_tokens - reasoning_output_tokens`, only
  when reasoning usage is reported.
- `total_tokens = input_tokens + output_tokens`; reasoning is not added again.

Missing fields remain null/unknown, distinct from an explicitly reported zero.
Each report gives the known sum and the number of calls with/without each
field. If cache detail is missing, the ordinary estimate explicitly lists its
zero-cache assumptions; the conservative guard continues to use upper rates.
Reported cache writes are charged at their separate standard tariff. Both the
price table/date and long-context multipliers are pinned; these figures are
**API-equivalent estimates, not subscription charges or remaining Codex quota**.
The account can also be used by this orchestrator or other sessions, so an
account-wide quota change cannot be attributed solely to an Astra experiment.

The [CLI JSON event receipt](https://learn.chatgpt.com/docs/non-interactive-mode)
provides a second usage check when available. It is compared with the provider
receipt, never added as a second charge. Known usage from an incomplete/failed
terminal response still counts even though its proposed code is not accepted.
A lost stream with no usable usage keeps its full reservation. Completed
responses are replayed idempotently, including from verified gzip archives.

After preparation, each evaluated iteration, and normal completion/failure,
the runner writes `reports/llm_usage/summary.json`, `summary.md` and `calls.csv`.
These contain **no prompt/answer text or host account/session identifiers** and
separate proposal/review roles and iterations. The JSON includes request/response
hash checks and missing-evidence warnings. To inspect a live run without writes
or model/account calls:

```sh
python -m tools.chia_loop.codex_cli usage --root eval_out/chia/ASTRA_XHIGH_RUN
```

Add `--write` to refresh the three derived report files. An abrupt process or
machine loss can interrupt report generation; the per-attempt stream, reservation
and ledger are the recovery evidence. Declared no-human-intervention policy and
observed stop events do not constitute proof against a hostile host operator.

For the private iteration retrospective, use:

```sh
python -m tools.chia_loop.codex_cli retrospective --root eval_out/chia/ASTRA_XHIGH_RUN
```

Add `--write` to render `reports/retrospective/index.json` and `index.md`.
The runner also renders these after preparation, each recorded iteration, and
normal completion/failure. Raw or verified gzip evidence can be read without
restoring it or launching a model. Unlike the content-free usage report, this
report includes model explanations and exposed summary text; keep it private
until explicitly reviewed for publication. A hash-match value of null means
there is no comparable receipt, not that integrity was verified. Raw artifacts
and evaluator receipts remain authoritative.

## Gemini completion reporting

`tools.chia_loop.completion_report` can watch independent Gemini supervisors
and write a read-only comparison after both reach a terminal state. It includes
final scores only for completed, archive-verified runs. It never steers or
modifies their optimization. This is a **local report generator, not a chat
wake-up mechanism**. Scheduled chat follow-up requires an available product
scheduler or a reachable app-server notification/queue integration.
