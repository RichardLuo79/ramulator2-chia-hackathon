# Changes to CHIA

`session-hooks.patch` extends [CHIA](https://github.com/ucb-bar/chia) revision
`16c35e92aaaf9511c6453bf94cd5cf589698f4e3` (`chialoops==1.0.1`) with the hooks this
framework needs to run, checkpoint and resume LLM sessions. The vendored copy in
`chia-loop/vendor/chia` already includes it. `session-hooks.json` records the
patch and the checksums of every file before and after it, and
`model_sessions.create_session` checks the imported source against them, so a
campaign always resumes on the same CHIA code.

## What changes, and why

| Change | Builds on | Why |
| --- | --- | --- |
| `prompt_once` | CHIA result types, error classifiers, CLI parsers and native capture/restore | Each attempt must be a single call that the campaign controls, so the adapters' built-in retries are bypassed. Failed CLI calls now capture the updated state before classifying the error. |
| Codex error classification | CHIA's error types and native CLI events | The upstream classifier searched the generated text for numbers such as `400` and reported an interrupted stream as a bad request. It now reads the native failure events, falls back to stderr only at startup and recognizes disconnected streams. |
| Raw CLI output and process-runner hook | Existing command builders and output parsers | Evidence must be complete, and every process must run in a controlled environment with a deadline. The caller supplies the runner. |
| Explicit executable, temporary-directory and working-directory options | Existing CLI command builders and session format | Each role runs a specific executable in its own directories. Claude now takes an executable path, both CLIs take a private temporary directory, and Claude takes a working directory. |
| Session-file read/write hooks | CHIA's capture/restore sequence; Ramulator's no-follow file operations | Session files are read and written without following links an agent may have created. The hooks cover transcripts, state and Codex's final-message file. |
| Claude appended prompt and explicit session ID | Native `--append-system-prompt`, `--session-id` and `--resume` | The task instructions are added to Claude's own system prompt, and a restarted adapter resumes the same session. |
| Codex private home and resume configuration | Native session files and `exec resume` | Each role keeps its state in its own home. `exec resume` takes the sandbox setting as a configuration override. |
| MCP authentication for the native CLIs | Codex `bearer_token_env_var` and Claude header expansion | The CLIs receive the name of an environment variable that holds the tool token, and the caller sets its value at launch. Codex also takes an explicit MCP approval policy. |
| Codex state inventory | Codex SQLite/WAL databases and rollout files | Codex 0.153.4 keeps history, goals, memory and queued state in separate databases. With `capture_all_sessions=True` and a role-private home, all of them and the child rollouts are captured; login files never are. The upstream default is unchanged. |
| Vertex continuation and event callback | CHIA's MCP loop and Google SDK `Content` objects | Each `prompt()` continues the same history, and raw responses, usage, signatures, call IDs and tool results survive checkpoints and failures. |
| Vertex JSON schemas | MCP-generated schemas and SDK `parameters_json_schema` | Tool schemas are passed unmodified, so nested tool arguments keep their `$defs` and `$ref`. |
| Vertex generation settings and HTTP client hook | SDK `ThinkingConfig`, `HttpOptions` and MCP HTTP transport | Thinking settings, per-request accounting and per-role tool tokens can be set through CHIA's own loop. |
| Chat Completions session hooks | CHIA's `OpenAICompatLLM`, SDK, error types and MCP tool loop | The adapter keeps the full history including provider reasoning, records every request, and offers the same single-attempt, checkpoint and event hooks as Vertex, with configurable thinking, scoped MCP authentication, provider-default output length and unlimited tool iterations. |
| API context hook | CHIA's Vertex and Chat Completions tool loops | An optional `context_compactor(adapter, client, history, config)` runs before each request, once tools have settled. The framework uses it for ADK summaries; CHIA itself gains no ADK dependency. |

## How attempts behave

`prompt_once` returns a CHIA result with `error` and the captured native state
when the provider or process fails. A failure to restore or capture state raises
an exception. On cancellation it captures what it can and re-raises. The caller
saves the evidence and then decides whether to retry.

Vertex runs tools in CHIA's loop and reports each request, response and tool
event through the callback, keeping thought signatures and function-call IDs.
After a completed tool group and a failed provider call, the session resumes
without repeating the user message or the tools. If a tool's result is
unresolved, the session stops, because the tool may already have edited files
or started an evaluation. Chat Completions (DeepSeek) follows the same rules and
keeps `reasoning_content`.

## Anthropic Messages on Vertex

`vertex_claude` calls the Messages API through CHIA's `ClaudeCodeLLM(backend="api")`
path with the official `AsyncAnthropicVertex` client, with the same state, MCP, event and
single-attempt hooks as the other API adapters. `max_tool_iterations=None`
lifts the tool-loop cap. Messages keep thinking signatures, compaction blocks and
tool IDs. A failed streaming request keeps its partial snapshot as evidence. The
provider deadline covers only the provider call.

Compaction uses Anthropic's threshold API: the service returns compaction blocks,
which are sent back unchanged with the history. Usage counts each reported
sampling iteration once, including compaction, and every response is archived.

## Applying the patch

To apply the patch to a clean checkout of the pinned revision:

```sh
git -C /path/to/chia apply --check /path/to/artifact/chia-loop/src/ramulator_chia/framework/upstream/session-hooks.patch
git -C /path/to/chia apply /path/to/artifact/chia-loop/src/ramulator_chia/framework/upstream/session-hooks.patch
```

`prompt()` keeps its retrying interface for compatibility; the framework always
calls `prompt_once`.

## Native CLIs

The framework's MCP server is named `ramulator`, because Claude CLI reserves
`workspace`. Claude runs with `--setting-sources ""` and `--strict-mcp-config`,
and keeps its transcripts in `CLAUDE_CONFIG_DIR/projects/<escaped-working-directory>`.

Codex's database stores absolute rollout paths, so a resumed Codex session must
run with its home at the recorded path; the session settings check this before
each call.

References:
[Codex MCP authentication and approval](https://learn.chatgpt.com/docs/extend/mcp),
[Codex state directory](https://learn.chatgpt.com/docs/config-file/config-reference),
[Claude MCP configuration](https://code.claude.com/docs/en/mcp).

`framework.agent.NativeAgent` connects these adapters to the campaign phases and
compressed checkpoints.
