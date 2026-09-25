# CHIA provider adapters

The vendored adapters use [CHIA](https://github.com/ucb-bar/chia) revision
`16c35e92aaaf9511c6453bf94cd5cf589698f4e3` (`chialoops==1.0.1`).

`session-hooks.patch` contains the adapter changes; `session-hooks.json`
records their source hashes. Session creation checks the installed adapter
identity before starting or resuming a campaign.

## Interface

| Facility | Behavior |
|---|---|
| `prompt_once` | One provider attempt; the campaign harness owns bounded retries. |
| Native state | Capture and restore role-specific transcripts, tool IDs, and continuation checkpoints. |
| Process runner | Explicit executable, working directory, private state paths, and scoped environment. |
| Event callback | Record request, response, usage, and tool events as they occur. |
| Tool authentication | Pass credential environment-variable names; the trusted driver binds their values. |
| Context compaction | Run a configured compactor after tool effects settle and before the next provider request. |

Provider/process failures return an error and captured state. Capture or restore
failures raise infrastructure exceptions. Cancellation captures available state
and propagates the interrupt. An unresolved tool effect blocks automatic
continuation; a settled tool exchange is not replayed after a provider failure.

## API backends

Vertex retains Google SDK message blocks, opaque thought signatures, function
call IDs, and complete tool schemas. Chat Completions retains SDK messages,
including `reasoning_content`, and per-request usage.

The API compaction hook uses the application's separate ADK summarizer.
Credentials and tool access remain with the trusted driver. Provider deadlines
cover provider requests, not time spent waiting for evaluations.

The optional `vertex_claude` adapter uses Anthropic's official Vertex client.
It preserves thinking signatures, compaction blocks, and tool IDs. Compaction
uses the provider's threshold API and default summary instructions. Accounting
counts each reported sampling iteration once. This backend is not used by
the four published campaign templates.

## Native CLI backends

Each campaign role has its own native state directory. Session file operations
reject links outside that directory. Authentication files are not continuation
assets.

The MCP server name is `ramulator`. Claude receives
`--setting-sources ""` and `--strict-mcp-config`; transcripts use its native
`CLAUDE_CONFIG_DIR/projects/<escaped-working-directory>` layout.

Codex stores absolute rollout paths. Resume and migration must preserve the
recorded native home; database contents are not rewritten to relocate it.
Background work must be settled before moving a checkpoint. Error
classification uses native failure events and startup stderr, not successful
model output.

## Development

Normal setup installs the patched vendored package. To apply the patch to a
separate checkout at the pinned CHIA revision:

```sh
git -C /path/to/chia apply --check /path/to/artifact/chia-loop/src/ramulator_chia/framework/upstream/session-hooks.patch
git -C /path/to/chia apply /path/to/artifact/chia-loop/src/ramulator_chia/framework/upstream/session-hooks.patch
```

Run the scripted adapter tests from the artifact root:

```sh
.venv-campaign/bin/python -m pytest -q \
  chia-loop/tests/test_chia_framework_model_adapters.py \
  chia-loop/tests/test_chia_framework_deepseek.py \
  chia-loop/tests/test_chia_framework_agent.py
```

These tests use native-format fixtures without provider calls. Do not run
upstream live-service tests with production credentials. The campaign framework
uses `prompt_once`; the upstream-compatible `prompt()` retry interface is
not used for campaign attempts.
