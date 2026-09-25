# Provider setup

Only the trusted driver authenticates to providers. Keep credentials outside
the repository, campaign outputs, and archives, with owner-only permissions.
The agent gets a fresh allowlisted workspace and authenticated CHIA tools; it
cannot browse this checkout's published results or another campaign.

| Template | Backend | Model | Effort | Search schedule |
|---|---|---|---|---|
| `configs/astra.json` | Codex CLI, ChatGPT login | `gpt-6-astra` | `xhigh` | 10 single-core + 5 multicore |
| `configs/deepseek.json` | DeepSeek API | `deepseek-flash` | `max` | 10 + 5 |
| `configs/gemini.json` | Gemini on Vertex | `gemini-3.8-flash` | `high` | 10 + 5 |
| `configs/opus.json` | Claude Code, subscription token | `claude-opus-5-5` | `xhigh` | 10 single-core |

These are explicit experiment settings, not model aliases. Availability and
account entitlements can change. Never silently substitute a model or billing
route. The dated tariff files under `configs/tariffs/` estimate API-equivalent
usage; review rates before a fresh experiment. Subscription charges are not
inferred from those estimates.

## GPT-6 Astra: Codex CLI

Install the native Linux CLI using the
[official installation instructions](https://learn.chatgpt.com/docs/codex/cli).
Use a version compatible with the vendored CHIA adapter; record the output of
`codex --version` with your experiment. Point `CHIA_CODEX_BIN` at the native
ELF executable, not an npm launcher. Its `codex-code-mode-host` companion must
be in the same directory. The preflight checks both.

These instructions use Codex CLI **0.153.4**.

Authenticate using the [official ChatGPT login](https://learn.chatgpt.com/docs/auth):

```sh
codex login --device-auth
export CHIA_CODEX_BIN=/absolute/path/to/native/codex
export CHIA_CODEX_AUTH="$HOME/.codex/auth.json"
chmod 600 "$CHIA_CODEX_AUTH"
```

Use Codex's file credential store (`cli_auth_credentials_store = "file"`) if
your installation otherwise uses a keychain. The trusted relay uses this one
managed login and lets Codex handle refresh; no prior threads are imported.
Native Codex compaction remains enabled through the existing adapter.

## Opus 5.5: Claude Code

Use Anthropic's [native installer](https://code.claude.com/docs/en/setup).
Set `CHIA_CLAUDE_BIN` to its resolved ELF binary and record `claude --version`.
An npm wrapper is not supported by the protected profile.

These instructions use Claude Code **2.1.282**.
The native installer accepts a version argument; pin your installation and
record its binary hash before starting a campaign.

Run [`claude setup-token`](https://code.claude.com/docs/en/authentication#generate-a-long-lived-token)
interactively. Save only the resulting subscription token in a private file
outside this repository, such as `$HOME/.claude/chia-oauth-token`:

```sh
export CHIA_CLAUDE_BIN=/absolute/path/to/native/claude
export CHIA_CLAUDE_AUTH="$HOME/.claude/chia-oauth-token"
chmod 600 "$CHIA_CLAUDE_AUTH"
```

Do not put the token in a command-line argument, configuration JSON, or Git.
Confirm your account's extra-usage setting before inference; subscription
authentication alone does not guarantee that paid overages are disabled.
The relay does not fall back to API-key billing. Claude manages native
compaction and session history.

## DeepSeek V4.1 Flash: direct API

Set `DEEPSEEK_API_KEY` in the trusted driver's environment using your preferred
secret manager or a private shell session. Do not store it under the checkout
or echo it into logs. The pinned `openai` SDK is installed by campaign setup.
The adapter retains the configured model and effort and disables SDK retries;
CHIA owns the bounded retry policy.

## Gemini 3.8 Flash: Vertex API

Set `CHIA_VERTEX_PROJECT` to **your** enabled Vertex project. No project is
provided or created by this artifact. Authenticate locally through
[Google's ADC instructions](https://docs.cloud.google.com/docs/authentication/set-up-adc-local-dev-environment):

```sh
gcloud auth application-default login
export CHIA_VERTEX_PROJECT=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
chmod 600 "$GOOGLE_APPLICATION_CREDENTIALS"
```

The driver uses the pinned Google SDK and the `global` endpoint. Generation,
token counting, and compaction use the same project. The preflight checks the
local credential file without refreshing it or contacting Google. Billing,
permissions, model access, and quota must be configured by the account owner.

## API context compaction

Gemini and DeepSeek need a separate summarizer environment because CHIA and
ADK use different dependency versions:

```sh
python3 scripts/setup --component context
```

The launcher selects `configs/compaction/gemini.json` or `deepseek.json` and
resolves `.venv-context/bin/python`. Both use ADK 2.3.0, an 80% trigger, and eight
recent tool rounds. Gemini uses its native token count and a 1,048,576-token
capacity; DeepSeek retains the existing byte-bound trigger and 1,000,000-token
policy. The latter estimate alone is not an exact provider-capacity rejection.
The parent makes and accounts for the summary request using the same provider.
No credentials or tool access are given to the summarizer subprocess.

Use `--context-compaction path.json` for an explicitly recorded alternative;
resume requires the original policy. CLI backends use native compaction and
do not need this environment.

## Before inference

Run `scripts/run-campaign ... --check`. A successful check proves local setup,
not a successful live model/tool exchange. It does not authenticate, refresh
tokens, consume inference credits, or create a campaign.
Only a launch with `--allow-paid` permits model calls. Do not use an existing
campaign as a smoke test.
