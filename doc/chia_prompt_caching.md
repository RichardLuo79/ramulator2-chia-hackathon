# Prompt caching in the CHIA backends

These measurements and settings describe the legacy runners. The
[common framework](../tools/chia_loop/framework/README.md) preserves native
sessions within each iteration and has no `prefix_v1`/`legacy` layout switch.
The historical measurements below are not its cache-performance results.

The `prefix_v1` layout improves reuse of repeated input in Gemini, Codex/Astra,
and Claude/Fable. It is the default for **new preparations**, with an explicit
`legacy` layout for ablations. Existing campaigns, response archives, usage
ledgers, and launch holds are not modified. No paid calls were made to develop
or verify this change.

## Observed baseline

The following are individual-run measurements from the pre-fix rich-DDR5
campaigns, audited on 2026-09-07. Percentages are token-weighted cache reads,
not the fraction of requests that hit a cache or a subscription-quota measure.

| Individual run | Cached input / reported total input | Cache counters / input receipts | Attempts without input receipts |
| --- | ---: | ---: | ---: |
| Astra xhigh | 1.17% | 126 / 126 | 5 |
| Astra max | 1.95% | 123 / 123 | 4 |
| Fable xhigh | 3.33% | 16 / 16 | 3 |
| Fable max | 2.35% | 6 / 6 | 0 |
| Gemini 3.8 Flash | at least 85.92% | 399 / 469 | 1 |
| Gemini 3.1 Pro | at least 2.05%; incomplete cache reporting | 2 / 57 | 3 |

Run roots are under `eval_out/chia/`: Astra and Gemini use
`{astra_xhigh,astra_max,gemini_flash,gemini_pro}_rich_ddr5_20i_20260906`;
Fable uses `{fable_xhigh,fable_max}_rich_ddr5_20i_oauth_20260906`.
Each ledger is analyzed separately. Unreported cache counters are not assumed
to be zero, and attempts with unknown usage are excluded from the token
denominator and counted separately.

## Causes and changes

**Codex/Astra.** Each fresh native CLI process previously supplied a different
`prompt_cache_key` and embedded attempt-specific workspace paths before the
conversation. The broker now supplies a stable, run-scoped key, omits optional
native input IDs, and virtualizes only audited native path metadata. Source
text, diagnostic paths inside supplied data, and real filesystem permissions
are unchanged. Unknown native framing fails before dispatch rather than being
normalized broadly. Stable prefix content and routing keys are relevant to
OpenAI's documented [prompt-cache behavior](https://developers.openai.com/api/docs/guides/prompt-caching).

For the public Responses API path, the layout also sets explicit cache
breakpoints and a 30-minute TTL. The ChatGPT subscription path retains native
cache defaults: public API cache options are **not** assumed compatible with
the separate subscription endpoint. Its stable key and input layout are
tested offline; provider-side hit rates remain to be measured.

**Claude/Fable.** The wrapper supplied the entire growing conversation as one
JSON text block. A cache entry ending after that block's closing `]` could not
match the longer conversation on the next call. The new layout splits the
same serialized text into stable content blocks, with markers before the
mutable closing bracket. It retains the previous request's history boundary
explicitly, even when new diagnostic output is too long for automatic
lookback. One system marker and at most three conversation markers respect
the [documented four-marker limit](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).
Native billing attribution, authentication, adaptive thinking, and the
one-hour TTL are preserved. This addresses a wrapper-layout problem; ordinary
Claude Code already manages [native conversation caching](https://code.claude.com/docs/en/prompt-caching).

**Gemini.** Flash already exhibited substantial reuse. Its native append-only
contents, including opaque thought signatures, are preserved. A new initial
prompt template places invariant policy, training configuration, and baseline
information before iteration-dependent content. Both models retain implicit
caching; the harness does not create explicitly billed cache-storage
resources. Google's [cache documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview)
recommends stable common prefixes and documents a 90% input-price discount
for cache hits.

The CLI initial prompt receives the same stable-first ordering. All fields and
values remain present. Serialized conversation roles remain data inside the
existing JSON envelope; they are not promoted into new native instruction
roles. Concatenating the generated blocks reproduces that envelope exactly.
Fragments are 16,384 characters, a layout parameter rather than a content cap;
no source, history, or diagnostic output is truncated for caching.

## Isolation and reproducibility

- Cache scopes are distinct for each run, provider, exact model, effort, and
  proposer/reviewer role. A random run nonce and the resolved run root bind
  the scope. No candidate, interaction, or server-side conversation is
  imported from another experiment.
- CLI processes still use fresh isolated homes, explicit input, and disabled
  memories, project instructions, tools, and persistent sessions. Caching
  is not implemented through session resumption or `previous_response_id`.
- `prompt_cache.json`, prompt templates, and implementation hashes are frozen
  with the experiment. Deleting or changing a prepared policy fails closed.
  A cache-layout ablation requires a fresh campaign, not an unrecorded switch
  inside an existing scientific continuation.
- Stable cache scope does not guarantee cache residency. Expiry, provider
  routing, minimum cacheable lengths, changing source, and long evaluation
  gaps still matter. Native date metadata remains truthful; a date change
  can invalidate the following prefix. Reviews often share less input than
  successive diagnostic turns.
- Evaluation windows, train/test boundaries, transfer isolation, diagnostics,
  compliance checks, model effort, promotion rules, and CPU limits are unchanged.
  Prompt ordering and transport layout are nevertheless recorded protocol
  changes; no claim of identical stochastic model output is made.

## Usage and accounting

For calls with reported input, the token-weighted fraction is
`sum(cached input tokens) / sum(total input tokens)`.

| Provider | Total-input denominator | Cache-read numerator |
| --- | --- | --- |
| Codex | `input_tokens`, which already includes cached tokens | `input_tokens_details.cached_tokens` |
| Claude | `input_tokens + cache_creation_input_tokens + cache_read_input_tokens` | `cache_read_input_tokens` |
| Gemini | `prompt_token_count` | `cached_content_token_count` |

Reports include call coverage, missing counters, unknown usage, and separate
role/iteration summaries. If some reported inputs lack cache counters, the
exact aggregate fraction is `null`; known reads divided by reported input
give a lower bound. The fraction restricted to calls with cache counters is
also available, but that subset may be biased.

Native CLI usage reports include this summary. Raw provider receipts remain
the accounting source. Gemini's newly tagged ledger entries in both layout
modes price known cache reads at the same documented discount; missing counters retain an
uncached upper estimate. Historical estimates are not rewritten, and the
conservative spending guard is unchanged. Cache reuse does not make generated
reasoning/output cheaper or reveal how a subscription provider weights weekly
quota consumption.

## Operating and verifying

Gemini preparation and both CLI preparation/queue commands accept:

```sh
--prompt-cache prefix_v1
```

This is the default. `--prompt-cache legacy` selects the old request layout;
it does **not** disable all provider-side caching. Roots without the policy
file keep legacy behavior, but still require their own frozen runner version.
No existing manifest is re-fingerprinted automatically to permit an upgrade.

Read one run's counters without authentication, writes, or model calls:

```sh
python -m tools.chia_loop.prompt_cache --root eval_out/chia/RUN_ID
```

Offline regression coverage includes lossless long/Unicode conversations,
stable and distinct namespaces, policy mutation and continuation rejection,
unknown-counter accounting, Gemini history/signature preservation, and the
actual native CLIs against local fake providers. Both xhigh/max efforts are
tested, with ChatGPT/API Codex modes and native-login/setup-token Claude modes.
Compression/replay tests verify that a saved complete response is reused
without another provider request. Preparation includes these cache preflights.
Native context hashes are deliberately strict; a CLI update needs a new audit.

Actual post-fix cache performance requires an authorized measured campaign.
The separate legacy Fable transport repair adds live forwarding and waits for
the producer before reconciling its receipt, without accepting truncated
answers or altering effort. Neither change upgrades a frozen campaign in place.
Detailed rerun/recovery notes remain local; the
[research overview](chia_hackathon_review.md) reports the current experiment.
