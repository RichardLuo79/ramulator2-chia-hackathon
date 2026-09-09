# CHIA session extension

This is a local patch against [CHIA](https://github.com/ucb-bar/chia) revision
`16c35e92aaaf9511c6453bf94cd5cf589698f4e3` (`chialoops==1.0.1`). It has not been
submitted upstream. It does not replace CHIA with a second model runtime.

`session-hooks.json` records the patch and before/after file checksums. The
framework's `model_sessions.create_session` checks the imported source bytes,
not only the distribution's version. A different implementation requires a new
qualification record; it cannot silently enter a continued campaign.

## What changes, and why

| Extension | Existing facility retained | Demonstrated gap |
| --- | --- | --- |
| `prompt_once` | CHIA result types, error classifiers, CLI parsers and native capture/restore | A workflow-owned attempt cannot use the adapters' hidden retry loops. Failed CLI calls previously classified errors before capturing updated state. |
| Codex error classification | CHIA's error types and native CLI events | Upstream searched generated text for bare numbers such as `400`, misclassifying an interrupted stream as a bad request. The classifier now reads native failure events, uses stderr only as a startup fallback, and recognizes disconnected streams. Successful output is never searched for errors. |
| Raw CLI output and process-runner hook | Existing command builders and output parsers | Human-readable previews truncate evidence; inherited environments and unbounded streaming waits cannot enforce our launch contract. The caller supplies the scoped, deadline-enforcing runner. |
| Explicit native file and executable options | Existing CLI command builders, temporary files and session format | A hard-coded executable or global temporary path cannot fit the role's positive process grant. Codex already accepts its executable; Claude now does too. Both accept a private temporary directory, and Claude accepts an explicit working directory. |
| Session-file read/write hooks | CHIA's existing native file inventory and capture/restore sequence; Ramulator's no-follow file operations | A trusted reader must not follow an agent-created symlink or hardlink outside its session. Hooks cover transcript/state restore and capture, and Codex's final-message file. They do not parse or reconstruct native history. |
| Claude append prompt and explicit session ID | Native `--append-system-prompt`, `--session-id` and `--resume` | Replacing native guidance is an unintended treatment. A reconstructed adapter must recover the same private session rather than generate a different UUID. |
| Codex private home and resumed sandbox argument | Native session files and `exec resume` | The captured state must belong to one role. Installed CLI help confirms `exec resume` does not accept `--sandbox`; it accepts the equivalent native configuration override. |
| Native CLI MCP authentication | Codex `bearer_token_env_var` and Claude header environment expansion | The original command builders supplied only a URL. `tool_token_env(tool)` now supplies an environment-variable **name**, never the credential. The caller binds its value privately at launch. Codex also accepts an explicit native MCP approval policy, matching Claude's existing registered-tool preapproval. |
| Codex native state inventory | Original SQLite/WAL bytes and rollout files | Codex 0.153.4 stores conversation history, goals, memory and queued state in separate databases. `capture_all_sessions=True` includes those named stores and child rollouts only when the caller supplies a role-private home. The narrower upstream inventory remains the default. Login files are not continuation assets. |
| Vertex continuation and event callback | Existing CHIA MCP loop and Google SDK `Content` objects | A new `prompt()` previously started a new history. Raw responses, usage, signatures, call IDs and tool acknowledgements must survive checkpoints and failures. |
| Vertex JSON schemas | MCP-generated schemas and SDK `parameters_json_schema` | Removing `$defs` while retaining `$ref` breaks nested tool arguments. The SDK already supports the unmodified JSON schema. |
| Explicit Vertex generation settings and HTTP client hook | SDK `ThinkingConfig`, `HttpOptions` and MCP HTTP transport | Thinking settings, per-request accounting and role-token authentication must be expressible without replacing the provider/tool loop. |

The new single-attempt entry point returns a CHIA result on provider/process
failure, with `error` and the captured native state. Failure to restore or capture
state raises an infrastructure exception. Cancellation captures available state
and still propagates the interrupt. The caller must save the evidence and resolve
the attempt before deciding whether to retry.

Vertex still executes tools in CHIA's loop. Its callback exposes each request,
response and tool event separately. It preserves SDK serialization, including
opaque thought signatures and function-call IDs. An unresolved tool checkpoint
stops continuation: it is not permission to repeat a possibly completed edit or
evaluation. A completed tool group followed by a failed provider call can resume
without repeating either the original user message or the tools.

## Installation and offline checks

Use a separate checkout and environment. Keep the existing installed package and
historical campaign runtimes untouched while reviewing this patch. Given a clean
checkout at the pinned revision:

```sh
git -C /path/to/chia apply --check /path/to/ramulator2/tools/chia_loop/framework/upstream/session-hooks.patch
git -C /path/to/chia apply /path/to/ramulator2/tools/chia_loop/framework/upstream/session-hooks.patch
```

After installing the ordinary project requirements, either install that checkout
with normal pip tooling or put it first on `PYTHONPATH` for offline qualification:

```sh
PYTHONPATH=/path/to/chia:.:python python -m pytest -q \
  tests/unit_tests/test_chia_framework_model_adapters.py
```

These tests prohibit network connections and model subprocesses. They exercise
the actual adapter implementation with native-format fixtures and real Google
SDK/MCP schemas. They also exercise the shared strict settings binding, native
checkpoint checksums, cross-role rejection and failure-state restoration. They
do not establish real-model availability or native CLI process isolation.

The selected upstream adapter tests can be run with the same no-services fixture
registered globally. Explicitly exclude live/cluster tests; some upstream tests
otherwise enable paid calls when credentials happen to be available:

```sh
PYTHONPATH=/path/to/chia:.:python:tests/unit_tests python -m pytest -q \
  -p test_chia_framework_model_adapters \
  /path/to/chia/chia/models/tests/test_codex.py \
  /path/to/chia/chia/models/tests/test_claude_api.py \
  /path/to/chia/chia/models/tests/test_vertex.py -k 'not live and not cluster'
```

The patch is not an inference launcher. Process/credential/control-plane
isolation, per-provider request settlement, common-episode integration and native
CLI qualification remain separate acceptance requirements. Legacy `prompt()`
retains its retry interface for upstream compatibility; the replacement framework
uses `prompt_once` and must never silently fall back to that legacy retry path.

## Installed CLI qualification

Earlier checks of Codex 0.153.4 and Claude Code 2.1.263 exercised an authenticated
MCP call and native resume against a local scripted provider. Those superseded
checks are preserved with the old implementation. They are not evidence that
the replacement's production launch boundary is ready.

The common application uses the MCP server name `ramulator`: `workspace` is
reserved by this Claude CLI. Claude receives `--setting-sources ""` and
`--strict-mcp-config`, and its transcript location is the native
`CLAUDE_CONFIG_DIR/projects/<escaped-working-directory>` directory.

Codex's database contains absolute rollout paths. A replacement worker must
restore this profile at its recorded absolute home. The common settings identity
enforces that requirement before another call; the framework does not rewrite
native SQLite contents or omit a database to make relocation appear to work.
The original bytes, including child rollouts, remain available for examination.
Whole-agent-tree recovery with running background work still needs qualification.

These additions retain CHIA's command builders, parsers and server lifecycle.
The test provider is only an inert fixture hosted through FastMCP's standard
custom-route facility; it is not another production provider or tool protocol.
The runtime test reuses the existing port-restricted boundary and explicitly
does **not** treat a TCP-port rule as destination isolation.

Native configuration references:
[Codex MCP authentication and approval](https://learn.chatgpt.com/docs/extend/mcp),
[Codex state directory](https://learn.chatgpt.com/docs/config-file/config-reference),
and [Claude MCP configuration](https://code.claude.com/docs/en/mcp).

The replacement `framework.agent.NativeAgent` connects these adapters to the
common campaign phases and compressed native checkpoints. Offline tests exercise
the actual adapters with inert process/SDK transports. They cover continuous
proposer sessions, independent critique, summary handoff, raw failure evidence,
usage scope and bounded continuation. This does not certify an online launch
boundary. See the
[implementation status](../../../../doc/chia_framework_implementation.md)
for current evidence and known live-runtime limitations.
