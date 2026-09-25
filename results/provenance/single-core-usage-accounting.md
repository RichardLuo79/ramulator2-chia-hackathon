# Single-core campaign usage accounting

Usage covers development, review, reflection, compaction, and recorded retries
in rounds 1–10 of the four single-core campaigns.

## Token totals and estimated cost

| Campaign | Input tokens | Output tokens, including reasoning | Estimated API-equivalent USD | Accounting coverage |
|---|---:|---:|---:|---|
| GPT-6 Astra | 92,843,921 | 464,015 | $639.03 recorded subtotal | 825 recorded requests; two interrupted tails have unknown additional usage |
| DeepSeek V4.1 Flash | 195,628,966 | 3,048,189 | $7.98 recorded subtotal | 1,245 requests with usage; three failed requests lack counters |
| Gemini 3.8 Flash | 583,211,000 | 1,696,409 | $57.85–73.34 | Input/output available for all 2,017 recorded requests; 81 cache partitions unknown |
| Opus 5.5 | 252,819,200 | 3,346,076 | $209.87 | All 30 main-agent CLI invocations; subagent usage is not separately reported |

Estimates use the tariffs recorded on September 19, 2026 for GPT-6 Astra,
DeepSeek V4.1 Flash, and Gemini 3.8 Flash, and September 23 for Opus 5.5.
GPT-6 Astra and Opus 5.5 used subscriptions. Gemini credits are not deducted;
DeepSeek uses the recorded peak-rate assumption. These are API-equivalent
estimates, not invoices. They exclude subscription fees and infrastructure.

Input counts include repeated context and cache traffic. Output includes
reasoning tokens. Tokenizers differ across providers. A recorded subtotal
does not bound the complete campaign's cost when some usage is unknown.

## Counting rules

- **GPT-6 Astra:** use per-request `token_usage_record` counters, deduplicated
  by response identity and reconciled with turn and thread totals. Cumulative
  CLI totals are not added again. The 825 recorded requests are below the
  272,000-token context-tier threshold; the largest has 229,202 input tokens.
  Two interrupted invocations have unknown additional usage.
- **DeepSeek V4.1 Flash:** use provider-event counters, including 32 compaction
  calls, one format-only review repair, and known responses within failed
  attempts. Three failed requests lack counters. Reviews in rounds 4 and 10
  were skipped for unchanged candidates. Each response is counted once.
- **Gemini 3.8 Flash:** include all recorded generations, including the
  interrupted round-10 reflection and its continuation. Missing cache
  partitions in 81 responses produce a cost interval. There are no separately
  identified compaction requests in these ten rounds.
- **Opus 5.5:** use per-invocation `usage`, without adding cumulative
  resumed-session `modelUsage`. The 2,112,657 reported reasoning tokens are
  part of output, not an additional charge. All 11,841,397 cache-write tokens
  use the recorded one-hour cache-write rate.

Cost uses disjoint uncached-input, cache-read, cache-write, and output
categories. Round and phase tables sum the same accounting ledger.
Identifiable compaction is a separate phase only when it is not already
included in an invocation aggregate.

## Data and reproduction

The paper notebook's Figure 5 uses the usage data in
[`paper-additions.json`](../paper/paper-v1/paper-additions.json).
Run `scripts/reproduce-figures` from the repository root to regenerate it
offline. The accounting code is in
[`campaign_usage.py`](../../analyses/campaign_usage.py).

Detailed round, phase, retry, tariff, and invocation tables are under
[`results/tables/chia_campaign_results/`](../tables/chia_campaign_results/).
The tariff table records source URLs, dates, rates, context tiers, and
cache-write durations. Only counters, statuses, rates, and evidence hashes
are included; private session state and credentials are excluded.
