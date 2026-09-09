# Independent Fable 5.1 CHIA pipelines

This describes the historical runner. New campaigns use the
[common framework](../tools/chia_loop/framework/README.md). Its native tools,
continuous proposer sessions and same-effort advisory reviewer replace the
legacy restrictions and blocking-review policy below. Fable max is currently
on hold; see the [current status](chia_framework_implementation.md).

Two fresh experiments use the unmodified Claude Code executable with the exact
model `claude-fable-5-1`: one proposer at `xhigh`, the other at `max`. Both use an
isolated **Fable 5.1 xhigh** compliance reviewer and the existing six-rule rubric.
Reviewer calls belong to the proposing run's usage ledger. No Ultracode,
workflows, native tools, model fallback, or effort downgrade is permitted.

The requested ceiling is **20 evaluated designs per run**, with no USD stopping
guard. Rejected drafts do not consume evaluated iterations but all their model
calls are recorded. The 48-turn, 192-diagnostic, 12-draft, three-review-format
attempt limits and 24-hour operational guard remain. Profiles are configurable
before preparation and immutable afterward. Each effort has its own run root,
history, selected source, receipts and reports; comparison is operator-side.

## Scientific contract

Each run begins with the unchanged fixed-delay Atomic skeleton. Only the marked
model/include regions, including model-specific parameters, are editable.
Final departure is committed on admission, state is bounded and causal, and
there is no command scheduler, reference-model invocation, workload fingerprint,
per-workload calibration or frontend-specific branch. Approximations require a
physical explanation and stated limitations.

The protected evaluator, metrics and scientific search operations are reused
from the Astra implementation. An AST regression test verifies unchanged
prompt construction, compliance review, drafting, repair, search, promotion and
freeze-before-test behavior, allowing only provider labels to differ. Separate
entry points avoid modifying the protocol files of ongoing campaigns.

- SimpleO3 + DDR5: eight training and eight disjoint test families, 20 million
  issued instructions per core, cold prefix and complete drain. Exact cohorts:
  `tools/chia_loop/configs/ddr5_frontend_transfer_v1.json`.
- Closed-loop promotion minimizes core-cycle macro absolute percentage error
  and request macro MAE/L. Each workload's L is its mean oracle read latency.
  Neither objective may increase and at least one must decrease. Incomparable
  candidates can remain parents in the nondominated archive.
- Training statistics, logical/controller trace slices, paired extremes and
  agent-selected generic synthetic diagnostics are enabled by the default
  ablation profile. No independent non-CHIA findings or special cases are used.
- FixedLat, MD1, WMG1 and MESS comparisons remain. The under-review bank-level
  Anatomy model is excluded.
- Selected source freezes before held-out SimpleO3 and DDR5 transfer: six
  ChampSim workloads (2M warmup + 20M ROI) and twelve gem5 SE programs to exit.
  Transfer scores never enter feedback. Nonexact ChampSim request coverage must
  remain labeled diagnostic-only rather than a valid paired headline metric.
- Every build/evaluation uses `-O3`. Jobs acquire three slots each from the
  shared twelve-CPU pool and wait if unavailable.

Agents see only their own supplied parent/history and allowed training evidence.
Prior feasibility designs, the independent non-CHIA worktree, previous model
campaigns, Git history, operator discussions and held-out results are excluded.
The benchmark families have been used previously by researchers; these are
fresh agent runs, not a claim of a historically untouched benchmark.

## Native CLI isolation and authentication

The implementation is tested against Claude Code **2.1.263**. Every action is
a fresh nonpersistent process with a private temporary home, explicit system
prompt and explicit same-run JSON history. Host instructions, environment,
memories, skills, hooks, plugins, MCP servers, workflows and native tools are
disabled. Nothing resumes a native Claude session.

Landlock/seccomp is installed before the CLI starts. It permits only the pinned
binary, runtime files, that invocation's temporary workspace and **one explicit
read-only credential file**. Candidate C++ has a separate boundary and
cannot read that credential. Claude's bundled Bun runtime additionally needs
its own `/proc/<pid>/maps` inventory; no process memory, environment, descriptors,
other processes or broad `/proc` access is granted. Tests check absolute paths,
parent traversal, escaping symlinks, sockets and non-gateway network ports.

Authentication uses the user's subscription through the unmodified native CLI,
not an API-key replacement client. Policy v2 freezes an explicit credential
kind and file path with each run:

- `native_login` (default): a temporary read-only binding to the selected
  `.credentials.json`. Its recorded access-token expiry must be at least 2,000
  seconds away before each invocation. Official re-login renews this file;
  the adapter does not implement refresh.
- `setup_token`: one opaque subscription token in an owner-only regular file
  outside the repository. The process boundary reads the file and injects
  `CLAUDE_CODE_OAUTH_TOKEN` into the native CLI environment in memory only.
  The token is never copied into saved configuration, command arguments,
  prompts, logs or archived workspaces. The file is re-read for each invocation.

The official `claude setup-token` command creates a documented one-year,
inference-only subscription token. See [Claude authentication](https://code.claude.com/docs/en/authentication#generate-a-long-lived-token).
This opaque token's actual expiry and subscription tier are not inferred from
its file age or contents: local status records them as unknown, and the provider
remains authoritative. Expiry or revocation can still stop a run. A provider
401 is not retried automatically, and any uncertain usage remains accounted for.
Never paste a token into chat or place it in Git; configure only its private path.

Both modes scrub inherited credentials and expose neither the host Claude
directory nor other sessions. A shell export does not reconfigure a frozen
queue. The transport forwards only the native CLI's audited Messages request
to Anthropic's fixed endpoint and does not archive authorization headers.
There is no API billing fallback, automatic extra-usage activation, or claim
that subscription quotas are unlimited. The gateway/boundary assume a trusted
installed CLI, runtime and host kernel. Mode changes require a fresh preflight
and explicit run provenance; existing frozen campaigns are not silently upgraded.

The actual outgoing request is checked for exact model, both top-level and
turn-scoped effort, adaptive thinking, 128,000 maximum output tokens, no tools,
and the precise supplied conversation. The CLI adds pinned attribution/outcome
boilerplate, the current UTC date and its native turn budget; these additions
are explicitly validated and recorded, not broadly accepted as arbitrary
context. A CLI update or unexpected prompt addition requires a new preflight.

## Accounting, reconstruction and recovery

Each native dispatch reserves estimated usage before contacting the provider.
Unknown outcomes retain their reservation. Completed streams are durably saved
and replayable without another model call; input, response, owner and effort
bindings must agree. There are at most three transport attempts per operation.
No truncated answer, contradictory receipt or native tool response is accepted.

Records include the supplied prompts/history, actual request (excluding private
routing metadata), wire-request hash, raw SSE, CLI events, provider request ID,
token usage, cache-write/read breakdown, retries, diagnostics, drafts, source
hashes, review decisions, metrics and promotions. Provider-exposed thinking is
retained when returned; missing thinking is labeled missing. Opaque signatures
and redacted thinking are not decoded, and this is not hidden chain-of-thought
access. The ledger accounts for output tokens once, including thinking.

API-equivalent estimates use the dated Fable tariff in `claude_cli/usage.py`.
Anthropic's `input_tokens` excludes cache reads/writes, so those categories are
added separately. Unknown cache TTL is conservatively priced and labeled.
Missing counters are unknown, not free. These estimates are neither subscription
invoices nor measurements of remaining account quota.

Traces use the existing verified compressed workflow. Finalized large
interaction/log/profile artifacts are gzip-compressed with integrity receipts;
replay supports raw or compressed evidence. Free-space guards stop instead of
deleting results. All operational repairs/interventions must be recorded; no
human or orchestrator supplies model improvements during a campaign.

## Commands and launch gates

Use the project's existing Python environment. Prepare each effort separately:

```sh
/tmp/ramulator-chia-smoke-venv/bin/python -m tools.chia_loop.claude_cli.queue \
  --root eval_out/chia/fable_xhigh_rich_ddr5_20i_oauth_20260906 \
  --effort xhigh --max-iterations 20 --cpus 3 \
  --credential-kind setup_token \
  --auth-file /home/dev/.claude/chia-oauth-token --prepare-only
```

For the second independent job, use effort `max` and root
`eval_out/chia/fable_max_rich_ddr5_20i_oauth_20260906`. Preparation verifies optimized
runtime provenance, full-window native parity, loaded-candidate isolation,
input identities, the unchanged seed and native-CLI offline tests. It makes
**zero model calls**. A failure does not auto-launch or weaken a gate.

After preparation and explicit run authorization:

```sh
/tmp/ramulator-chia-smoke-venv/bin/python -m tools.chia_loop.claude_cli supervise \
  --root eval_out/chia/fable_xhigh_rich_ddr5_20i_oauth_20260906 --authorize-paid
```

Alternatively, replace the queue's `--prepare-only` with `--authorize-paid`
to start generation automatically after every preparation gate passes. Native
login mode remains available through `--credential-kind native_login` and the
explicit private `.credentials.json` path. Store setup-token files with mode
`600`; the token value must not appear in a command line or run configuration.

Use `--resume` only for an interrupted same run, never to reopen a frozen search.
The initial live proposal is the first counted model call; offline CLI checks
do not establish paid model availability. `STOP` halts a run;
`<run-root>.preparation_STOP` also stops queued preparation. `usage` and
`retrospective` subcommands read one run's records; `--write` generates reports.

No existing Gemini/Astra run, independent worktree, remote or branch history is
changed by this backend. Nothing is pushed automatically.
